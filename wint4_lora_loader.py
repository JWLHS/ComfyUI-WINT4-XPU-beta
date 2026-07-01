"""
wint4_lora_loader.py — WINT4 LoRA Loader node for ComfyUI.
"""
import logging
import torch
import folder_paths
import comfy.utils
from .wint4_lora_common import _normalize_layer_path, _auto_detect_format, _convert_bfl_to_standard

log = logging.getLogger("WINT4-LoRA")

class WINT4LoRALoader:
	NAME = "WINT4 LoRA Loader"
	CATEGORY = "WINT4"

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model": ("MODEL", {"tooltip": "Model from WINT4ModelLoader"}),
				"lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA safetensors file"}),
				"strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
										"tooltip": "LoRA strength. 1.5-2.0× recommended for INT4."}),
			},
		}

	RETURN_TYPES = ("MODEL",)
	RETURN_NAMES = ("model",)
	FUNCTION = "load_lora"

	def load_lora(self, model, lora_name: str, strength: float):
		if abs(strength) < 1e-5:
			return (model,)

		lora_path = folder_paths.get_full_path("loras", lora_name)
		if lora_path is None:
			raise FileNotFoundError(f"[WINT4 LoRA] LoRA '{lora_name}' not found.")

		log.info(f"[WINT4 LoRA] Loading: {lora_name} (strength={strength})")

		diffusion_model = model.model.diffusion_model
		if getattr(model.model, '_lora_needs_reset', False):
			for module in diffusion_model.modules():
				if hasattr(module, '_lora_entries'):
					object.__setattr__(module, '_lora_entries', {})
				bake_state = getattr(module, '_wint4_bake_state', None)
				if bake_state is not None and '_orig_weight' in bake_state:
					module.weight.data.copy_(bake_state['_orig_weight'])
				object.__setattr__(module, '_wint4_bake_state', None)
			object.__setattr__(model.model, '_wint4_loras', [])
			object.__setattr__(model.model, '_lora_needs_reset', False)

		lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

		fmt = _auto_detect_format(lora_sd)
		if fmt == "bfl":
			lora_sd = _convert_bfl_to_standard(lora_sd)
			log.info(f"[WINT4 LoRA] Converted BFL → standard format")

		if hasattr(torch, "xpu") and torch.xpu.is_available():
			dev = torch.device("xpu")
		elif torch.cuda.is_available():
			dev = torch.device("cuda")
		else:
			dev = torch.device("cpu")

		lora_data: dict[str, dict] = {}
		is_lokr = False
		for key, tensor in lora_sd.items():
			if "lokr_w1" in key:
				is_lokr = True
				idx = key.index("lokr_w1")
				lp = key[:idx].rstrip(".")
				lp = _normalize_layer_path(lp)
				if lp is None: continue
				lora_data.setdefault(lp, {})["lokr_w1"] = tensor
			elif "lokr_w2" in key:
				idx = key.index("lokr_w2")
				lp = key[:idx].rstrip(".")
				lp = _normalize_layer_path(lp)
				if lp is None: continue
				lora_data.setdefault(lp, {})["lokr_w2"] = tensor
			elif "lora_up" in key or "lora_B" in key:
				idx = key.index("lora_up") if "lora_up" in key else key.index("lora_B")
				lp = key[:idx].rstrip(".")
				lp = _normalize_layer_path(lp)
				if lp is None: continue
				lora_data.setdefault(lp, {})["up"] = tensor
			elif "lora_down" in key or "lora_A" in key:
				idx = key.index("lora_down") if "lora_down" in key else key.index("lora_A")
				lp = key[:idx].rstrip(".")
				lp = _normalize_layer_path(lp)
				if lp is None: continue
				lora_data.setdefault(lp, {})["down"] = tensor
			elif key.endswith(".alpha"):
				lp = key[:-len(".alpha")]
				lp = _normalize_layer_path(lp)
				if lp is None: continue
				lora_data.setdefault(lp, {})["alpha"] = float(tensor.mean()) if tensor.numel() > 1 else tensor.item()

		if is_lokr:
			log.info(f"[WINT4 LoRA] Detected LyCORIS LoKr format")

		applied_quantized = 0
		applied_bake = 0

		for mod_name, module in diffusion_model.named_modules():
			norm_name = _normalize_layer_path(mod_name)
			if norm_name is None:
				continue

			is_quantized = getattr(module, '_is_quantized', False)

			candidates = []
			# QKV fusion fallback
			if norm_name.endswith(".attn.qkv") and hasattr(module, 'weight'):
				out_f = module.weight.shape[0]
				hs = out_f // 3
				if hs * 3 == out_f:
					for suffix, sl_start, sl_end in [
						(".attn.wq", 0, hs), (".attn.wk", hs, 2*hs), (".attn.wv", 2*hs, 3*hs),
					]:
						qkv_key = norm_name.replace(".attn.qkv", suffix)
						info = lora_data.get(qkv_key)
						if info is not None:
							candidates.append((info, sl_start, sl_end, qkv_key))

			info = lora_data.get(norm_name)
			if info is not None:
				candidates.append((info, None, None, norm_name))

			for info, sl_start, sl_end, lp_key in candidates:
				# ── LoKr (quantized only) ────────────
				if "lokr_w1" in info and "lokr_w2" in info:
					if not is_quantized:
						continue
					w1 = info["lokr_w1"]
					w2 = info["lokr_w2"]
					factor = w1.shape[0]
					alpha = info.get("alpha", factor)
					multiplier = alpha / max(factor, 1) * strength

					w1 = w1.to(dev, dtype=torch.float16)
					w2 = w2.to(dev, dtype=torch.float16)

					if getattr(module, '_use_quarot', False):
						H = getattr(module, '_hadamard_H', None)
						gs = getattr(module, '_group_size', 128)
						if H is not None and gs > 0 and w2.shape[1] % gs == 0:
							H_dev = H.to(dev, dtype=torch.float16)
							n_groups = w2.shape[1] // gs
							w2 = (w2.reshape(w2.shape[0], n_groups, gs) @ H_dev.T).reshape(w2.shape[0], w2.shape[1])

					lora_entries = getattr(module, '_lora_entries', None)
					if lora_entries is None:
						lora_entries = {}
						object.__setattr__(module, '_lora_entries', lora_entries)

					lora_entries.pop(lora_name, None)
					entry = ("lokr", w1, w2, multiplier, factor) if sl_start is None else ("lokr", w1, w2, multiplier, factor, sl_start, sl_end)
					lora_entries[lora_name] = [entry]
					applied_quantized += 1
					continue

				# ── Standard LoRA ─────────────────────
				up, down = info.get("up"), info.get("down")
				if up is None or down is None:
					continue
				rank = up.shape[1]
				alpha = info.get("alpha", rank)
				multiplier = alpha / max(rank, 1) * strength

				# ── Bake-in (non-quantized) ───────────
				if not is_quantized:
					if not hasattr(module, 'weight') or module.weight is None:
						continue
					w = module.weight

					A = down.to(dev, dtype=torch.float16, non_blocking=True)
					B = up.to(dev, dtype=torch.float16, non_blocking=True)

					if getattr(module, '_use_quarot', False):
						H = getattr(module, '_hadamard_H', None)
						gs = getattr(module, '_group_size', 128)
						if H is not None and gs > 0 and A.shape[1] % gs == 0:
							H_dev = H.to(dev, dtype=torch.float16)
							n_groups = A.shape[1] // gs
							A = (A.reshape(A.shape[0], n_groups, gs) @ H_dev.T).reshape(A.shape[0], A.shape[1])

					delta = (B @ A).mul_(multiplier)
					if sl_start is not None:
						delta = delta[sl_start:sl_end, :]

					bake_state = getattr(module, '_wint4_bake_state', None)
					if bake_state is None:
						bake_state = {'_orig_weight': w.data.clone()}
						object.__setattr__(module, '_wint4_bake_state', bake_state)

					bake_state[lora_name] = {'delta': delta.clone(), 'multiplier': multiplier}

					# 恢复 orig + 重新施加所有活跃 LoRA
					active = {e["name"] for e in getattr(model.model, '_wint4_loras', [])}
					active.add(lora_name)
					w.data.copy_(bake_state['_orig_weight'])
					for lora_n, info_b in bake_state.items():
						if lora_n == '_orig_weight':
							continue
						if lora_n in active:
							w.data.add_(info_b['delta'].to(device=w.device))

					applied_bake += 1
					continue

				# ── Quantized ─────────────────────────
				A = down.to(dev, dtype=torch.float16, non_blocking=True)
				B = up.to(dev, dtype=torch.float16, non_blocking=True)

				if getattr(module, '_use_quarot', False):
					H = getattr(module, '_hadamard_H', None)
					gs = getattr(module, '_group_size', 128)
					if H is not None and gs > 0 and A.shape[1] % gs == 0:
						H_dev = H.to(dev, dtype=torch.float16)
						n_groups = A.shape[1] // gs
						A = (A.reshape(A.shape[0], n_groups, gs) @ H_dev.T).reshape(A.shape[0], A.shape[1])

				lora_entries = getattr(module, '_lora_entries', None)
				if lora_entries is None:
					lora_entries = {}
					object.__setattr__(module, '_lora_entries', lora_entries)

				lora_entries.pop(lora_name, None)
				entry = (A, B, multiplier) if sl_start is None else (A, B, multiplier, sl_start, sl_end)
				lora_entries[lora_name] = [entry]
				applied_quantized += 1

		del lora_sd, lora_data

		if not hasattr(model.model, '_wint4_loras'):
			object.__setattr__(model.model, '_wint4_loras', [])
		model.model._wint4_loras.append({"name": lora_name, "strength": strength, "path": lora_path})

		# ── Prune ────────────────────────────────────
		active_names = {entry["name"] for entry in model.model._wint4_loras}
		for module in diffusion_model.modules():
			entries = getattr(module, '_lora_entries', None)
			if entries:
				for stale in list(entries.keys()):
					if stale not in active_names:
						del entries[stale]
			bake_state = getattr(module, '_wint4_bake_state', None)
			if bake_state is not None:
				for stale in list(bake_state.keys()):
					if stale == '_orig_weight':
						continue
					if stale not in active_names:
						del bake_state[stale]
				if len(bake_state) == 1 and '_orig_weight' in bake_state:
					module.weight.data.copy_(bake_state['_orig_weight'])
					object.__setattr__(module, '_wint4_bake_state', None)

		if applied_quantized > 0 or applied_bake > 0:
			log.info(f"[WINT4 LoRA] ✓ Loaded: {lora_name} → {applied_quantized} INT4 + {applied_bake} bake-in layers")
		else:
			log.warning(f"[WINT4 LoRA] ✗ NOT applied: {lora_name} — 0 layers matched (format: {fmt})")

		return (model,)


NODE_CLASS_MAPPINGS = {"WINT4LoRALoader": WINT4LoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4LoRALoader": "WINT4 LoRA Loader"}
