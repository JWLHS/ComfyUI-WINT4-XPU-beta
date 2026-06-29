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
        for key, tensor in lora_sd.items():
            if "lora_up" in key or "lora_B" in key:
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

        diffusion_model = model.model.diffusion_model
        applied = 0

        for mod_name, module in diffusion_model.named_modules():
            if not getattr(module, '_is_quantized', False):
                continue
            norm_name = _normalize_layer_path(mod_name)
            if norm_name is None:
                continue

            candidates = []

            # QKV fusion fallback
            if norm_name.endswith(".attn.qkv"):
                out_f = module.weight.shape[0]
                hs = out_f // 3
                if hs * 3 == out_f:
                    for suffix, sl_start, sl_end in [
                        (".attn.wq", 0, hs), (".attn.wk", hs, 2*hs), (".attn.wv", 2*hs, 3*hs),
                    ]:
                        qkv_key = norm_name.replace(".attn.qkv", suffix)
                        info = lora_data.get(qkv_key)
                        if info is not None:
                            candidates.append((info, sl_start, sl_end))

            info = lora_data.get(norm_name)
            if info is not None:
                candidates.append((info, None, None))

            for info, sl_start, sl_end in candidates:
                up, down = info.get("up"), info.get("down")
                if up is None or down is None: continue
                rank = up.shape[1]
                alpha = info.get("alpha", rank)
                multiplier = alpha / max(rank, 1) * strength

                A = down.to(dev, dtype=torch.float16, non_blocking=True)
                B = up.to(dev, dtype=torch.float16, non_blocking=True)

                # QuaRot A rotation
                if getattr(module, '_use_quarot', False):
                    H = getattr(module, '_hadamard_H', None)
                    gs = getattr(module, '_group_size', 128)
                    if H is not None and gs > 0 and A.shape[1] % gs == 0:
                        H_dev = H.to(dev, dtype=torch.float16)
                        n_groups = A.shape[1] // gs
                        A = (A.reshape(A.shape[0], n_groups, gs) @ H_dev.T).reshape(A.shape[0], A.shape[1])

                # ── lora_name-keyed dict ──────────────────────────
                lora_entries = getattr(module, '_lora_entries', None)
                if lora_entries is None:
                    lora_entries = {}
                    object.__setattr__(module, '_lora_entries', lora_entries)

                # Clear old entries for this lora_name
                lora_entries.pop(lora_name, None)

                entry = (A, B, multiplier) if sl_start is None else (A, B, multiplier, sl_start, sl_end)
                lora_entries[lora_name] = [entry]
                applied += 1

        del lora_sd, lora_data

        if not hasattr(model.model, '_wint4_loras'):
            object.__setattr__(model.model, '_wint4_loras', [])
        model.model._wint4_loras.append({"name": lora_name, "strength": strength, "path": lora_path})

        if applied > 0:
            log.info(f"[WINT4 LoRA] ✓ Loaded: {lora_name} → {applied} INT4 layers")
        else:
            log.warning(f"[WINT4 LoRA] ✗ NOT applied: {lora_name} — 0 INT4 layers matched (format: {fmt})")

        return (model,)

NODE_CLASS_MAPPINGS = {"WINT4LoRALoader": WINT4LoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4LoRALoader": "WINT4 LoRA Loader"}
