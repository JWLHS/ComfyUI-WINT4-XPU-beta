"""
wint4_lora_stack.py
───────────────────
WINT4 LoRA Stack node — apply multiple LoRAs to INT4 model at once.

Stores raw LoRA matrices (A=down, B=up) on XPU — NOT pre-expanded delta.
Forward computes delta = B @ A on-the-fly.
"""

import logging
import torch
import folder_paths
import comfy.utils

from .wint4_lora_common import (
    _normalize_layer_path,
    _auto_detect_format,
    _convert_bfl_to_standard,
)

log = logging.getLogger("WINT4-LoRA-Stack")


class WINT4LoRAStack:

    NAME = "WINT4 LoRA Stack"
    CATEGORY = "WINT4"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from WINT4ModelLoader"}),
            },
            "optional": {},
        }
        for i in range(1, 6):
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + folder_paths.get_filename_list("loras"),
                {"tooltip": f"LoRA {i}. 'None' to skip."},
            )
            inputs["optional"][f"strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                 "tooltip": f"Strength for LoRA {i}. 1.5-2.0× recommended for INT4."},
            )
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

        # ── Resolve storage device ────────────────────────────────
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        diffusion_model = model.model.diffusion_model
        total_applied = 0

        for lora_name, lora_path, strength in to_apply:
            log.info(f"[WINT4 LoRA Stack] Loading: {lora_name} (strength={strength})")
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)

            fmt = _auto_detect_format(lora_sd)
            if fmt == "bfl":
                lora_sd = _convert_bfl_to_standard(lora_sd)
                log.info(f"[WINT4 LoRA Stack] Converted BFL → standard")

            lora_data: dict[str, dict] = {}
            for key, tensor in lora_sd.items():
                if "lora_up" in key or "lora_B" in key:
                    idx = key.index("lora_up") if "lora_up" in key else key.index("lora_B")
                    lp = key[:idx].rstrip(".")
                    lp = _normalize_layer_path(lp)
                    if lp is None:
                        continue
                    lora_data.setdefault(lp, {})["up"] = tensor
                elif "lora_down" in key or "lora_A" in key:
                    idx = key.index("lora_down") if "lora_down" in key else key.index("lora_A")
                    lp = key[:idx].rstrip(".")
                    lp = _normalize_layer_path(lp)
                    if lp is None:
                        continue
                    lora_data.setdefault(lp, {})["down"] = tensor
                elif key.endswith(".alpha"):
                    lp = key[:-len(".alpha")]
                    lp = _normalize_layer_path(lp)
                    if lp is None:
                        continue
                    lora_data.setdefault(lp, {})["alpha"] = (
                        tensor.item() if tensor.numel() == 1 else float(tensor.mean())
                    )

            layer_applied = 0
            for mod_name, module in diffusion_model.named_modules():
                if not getattr(module, '_is_quantized', False):
                    continue
                full = f"diffusion_model.{mod_name}"
                info = lora_data.get(full)
                if info is None or "up" not in info or "down" not in info:
                    continue
                up, down = info["up"], info["down"]
                rank = up.shape[1]
                alpha = info.get("alpha", rank)
                multiplier = alpha / max(rank, 1) * strength

                A = down.to(dev, dtype=torch.float16, non_blocking=True)
                B = up.to(dev, dtype=torch.float16, non_blocking=True)

                entries = getattr(module, '_lora_entries', None)
                if entries is None:
                    entries = []
                    object.__setattr__(module, '_lora_entries', entries)
                entries.append((A, B, multiplier))
                layer_applied += 1
                total_applied += 1

            del lora_sd
            del lora_data

            if layer_applied > 0:
                log.info(f"[WINT4 LoRA Stack] ✓ Loaded: {lora_name} → {layer_applied} layers")
            else:
                log.warning(f"[WINT4 LoRA Stack] ✗ NOT applied: {lora_name} — 0 layers matched (format: {fmt})")

        log.info(f"[WINT4 LoRA Stack] Total: {total_applied} entries across {len(to_apply)} LoRAs.")
        return (model,)


NODE_CLASS_MAPPINGS = {"WINT4LoRAStack": WINT4LoRAStack}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4LoRAStack": "WINT4 LoRA Stack"}
