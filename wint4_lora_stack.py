"""
wint4_lora_stack.py — WINT4 LoRA Stack node for ComfyUI.
"""
import logging
import torch
import folder_paths
import comfy.utils
from .wint4_lora_common import _normalize_layer_path, _auto_detect_format, _convert_bfl_to_standard

log = logging.getLogger("WINT4-LoRA-Stack")

class WINT4LoRAStack:
    NAME = "WINT4 LoRA Stack"
    CATEGORY = "WINT4"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {"model": ("MODEL", {"tooltip": "Model from WINT4ModelLoader"})},
            "optional": {},
        }
        for i in range(1, 6):
            inputs["optional"][f"lora_name_{i}"] = (["None"] + folder_paths.get_filename_list("loras"), {"tooltip": f"LoRA {i}"})
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    def apply(self, model, **kwargs):
        to_apply = []
        for i in range(1, 6):
            name = kwargs.get(f"lora_name_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)
            if name is None or name == "None" or name == "" or abs(strength) < 1e-5:
                continue
            path = folder_paths.get_full_path("loras", name)
            if path is None:
                log.warning(f"[WINT4 LoRA Stack] LoRA '{name}' not found, skipping.")
                continue
            to_apply.append((name, path, strength))

        if not to_apply:
            return (model,)

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        diffusion_model = model.model.diffusion_model

        # ── Stack replaces everything ──────────────────────────
        for module in diffusion_model.modules():
            if hasattr(module, '_lora_entries'):
                object.__setattr__(module, '_lora_entries', {})
        object.__setattr__(model.model, '_wint4_loras', [])

        total_applied = 0

        for lora_name, lora_path, strength in to_apply:
            log.info(f"[WINT4 LoRA Stack] Loading: {lora_name} (strength={strength})")
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

            fmt = _auto_detect_format(lora_sd)
            if fmt == "bfl":
                lora_sd = _convert_bfl_to_standard(lora_sd)
                log.info(f"[WINT4 LoRA Stack] Converted BFL → standard")

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
                    lora_data.setdefault(lp, {})["alpha"] = (tensor.item() if tensor.numel() == 1 else float(tensor.mean()))

            if is_lokr:
                log.info(f"[WINT4 LoRA Stack] Detected LyCORIS LoKr format: {lora_name}")

            layer_applied = 0
            for mod_name, module in diffusion_model.named_modules():
                if not getattr(module, '_is_quantized', False):
                    continue
                norm_name = _normalize_layer_path(mod_name)
                if norm_name is None: continue

                candidates = []
                if norm_name.endswith(".attn.qkv"):
                    out_f = module.weight.shape[0]
                    hs = out_f // 3
                    if hs * 3 == out_f:
                        for suffix, sl_start, sl_end in [
                            (".attn.wq", 0, hs), (".attn.wk", hs, 2*hs), (".attn.wv", 2*hs, 3*hs),
                        ]:
                            qkv_key = norm_name.replace(".attn.qkv", suffix)
                            info = lora_data.get(qkv_key)
                            if info is not None: candidates.append((info, sl_start, sl_end))

                info = lora_data.get(norm_name)
                if info is not None: candidates.append((info, None, None))

                for info, sl_start, sl_end in candidates:
                    # ── LoKr path ──────────────────────────────
                    if "lokr_w1" in info and "lokr_w2" in info:
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

                        if sl_start is None:
                            entry = ("lokr", w1, w2, multiplier, factor)
                        else:
                            entry = ("lokr", w1, w2, multiplier, factor, sl_start, sl_end)
                        lora_entries.setdefault(lora_name, []).append(entry)
                        layer_applied += 1
                        total_applied += 1
                        continue

                    # ── Standard LoRA path ─────────────────────
                    up, down = info.get("up"), info.get("down")
                    if up is None or down is None: continue
                    rank = up.shape[1]
                    alpha = info.get("alpha", rank)
                    multiplier = alpha / max(rank, 1) * strength

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

                    entry = (A, B, multiplier) if sl_start is None else (A, B, multiplier, sl_start, sl_end)
                    lora_entries.setdefault(lora_name, []).append(entry)
                    layer_applied += 1
                    total_applied += 1

            del lora_sd, lora_data

            model.model._wint4_loras.append({"name": lora_name, "strength": strength, "path": lora_path})

            if layer_applied > 0:
                log.info(f"[WINT4 LoRA Stack] ✓ Loaded: {lora_name} → {layer_applied} layers")
            else:
                log.warning(f"[WINT4 LoRA Stack] ✗ NOT applied: {lora_name} — 0 layers matched (format: {fmt})")

        log.info(f"[WINT4 LoRA Stack] Total: {total_applied} entries across {len(to_apply)} LoRAs.")
        return (model,)


NODE_CLASS_MAPPINGS = {"WINT4LoRAStack": WINT4LoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4LoRAStack": "WINT4 LoRA Stack"}
