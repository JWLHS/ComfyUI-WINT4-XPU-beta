"""
wint4_lora_loader.py
────────────────────
WINT4 LoRA Loader node for ComfyUI.

Bypasses ComfyUI's native load_lora() entirely.
Reads LoRA safetensors directly, stores LoRA entries on INT4 modules.

Stores raw LoRA matrices (A=down, B=up) on XPU — NOT pre-expanded delta.
Forward computes delta = B @ A on-the-fly on XPU.

Supports:
  - Single LoRA (one WINT4LoRALoader node)
  - Chained LoRAs (multiple WINT4LoRALoader nodes in series)
  - Multi LoRA (WINT4LoRAStack node)

  LoRA key formats (auto-normalized via wint4_lora_common):
    ① diffusion_model.blocks.X.attn.wq.lora_B.weight          Kohya standard
    ② transformer.blocks.X.attn.to_q.lora_B.weight            diffusers/simpletrainer
    ③ lycoris_blocks_X_attn_wq.lora_down.weight               SimpleTuner lycoris
    ④ blocks.X.attn.wq.lora_B.weight                           bare (no prefix)
    ⑤ transformer.text_fusion.layerwise_blocks.X.attn.to_q... onetrainer
    ⑥ lora_unet_blocks_X_attn_wq.lora_down.weight             legacy ComfyUI
    ⑦ lora_transformer_blocks_X_attn_wq.lora_down.weight      onetrainer alt
    ⑧ single_blocks.X.attn.qkv.lora_A.weight                  BFL → auto-converted
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

log = logging.getLogger("WINT4-LoRA")

NODE_NAME = "WINT4 LoRA Loader"


class WINT4LoRALoader:

    NAME = NODE_NAME
    CATEGORY = "WINT4"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from WINT4ModelLoader"}),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "LoRA safetensors file"},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                     "tooltip": "LoRA strength. 1.5-2.0× recommended for INT4."},
                ),
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

        # ── Resolve storage device ────────────────────────────────
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        # ── Parse LoRA keys ───────────────────────────────────────
        lora_data: dict[str, dict] = {}
        for key, tensor in lora_sd.items():
            if "lora_up" in key or "lora_B" in key:
                idx = key.index("lora_up") if "lora_up" in key else key.index("lora_B")
                layer_path = key[:idx].rstrip(".")
                layer_path = _normalize_layer_path(layer_path)
                if layer_path is None:
                    continue
                lora_data.setdefault(layer_path, {})["up"] = tensor
            elif "lora_down" in key or "lora_A" in key:
                idx = key.index("lora_down") if "lora_down" in key else key.index("lora_A")
                layer_path = key[:idx].rstrip(".")
                layer_path = _normalize_layer_path(layer_path)
                if layer_path is None:
                    continue
                lora_data.setdefault(layer_path, {})["down"] = tensor
            elif key.endswith(".alpha"):
                layer_path = key[:-len(".alpha")]
                layer_path = _normalize_layer_path(layer_path)
                if layer_path is None:
                    continue
                alpha_val = tensor.item() if tensor.numel() == 1 else float(tensor.mean())
                lora_data.setdefault(layer_path, {})["alpha"] = alpha_val

        # ── Match & attach (store raw A/B on XPU) ─────────────────
        diffusion_model = model.model.diffusion_model
        applied = 0

        for mod_name, module in diffusion_model.named_modules():
            if not getattr(module, '_is_quantized', False):
                continue
            full_name = f"diffusion_model.{mod_name}"
            if full_name not in lora_data:
                continue

            info = lora_data[full_name]
            up = info.get("up")      # lora_B: (out_f, rank)
            down = info.get("down")  # lora_A: (rank, in_f)
            if up is None or down is None:
                continue

            alpha = info.get("alpha", up.shape[1])
            rank = up.shape[1]
            multiplier = alpha / max(rank, 1) * strength

            # Store A/B on XPU (fp16).  Forward uses them directly.
            A = down.to(dev, dtype=torch.float16, non_blocking=True)
            B = up.to(dev, dtype=torch.float16, non_blocking=True)

            entries = getattr(module, '_lora_entries', None)
            if entries is None:
                entries = []
                object.__setattr__(module, '_lora_entries', entries)
            entries.append((A, B, multiplier))
            applied += 1

        # Release raw safetensors data
        del lora_sd
        del lora_data

        if not hasattr(model.model, '_wint4_loras'):
            object.__setattr__(model.model, '_wint4_loras', [])
        model.model._wint4_loras.append({
            "name": lora_name,
            "strength": strength,
            "path": lora_path,
        })

        if applied > 0:
            log.info(f"[WINT4 LoRA] ✓ Loaded: {lora_name} → {applied} INT4 layers")
        else:
            log.warning(f"[WINT4 LoRA] ✗ NOT applied: {lora_name} — 0 INT4 layers matched (format: {fmt})")

        return (model,)


NODE_CLASS_MAPPINGS = {"WINT4LoRALoader": WINT4LoRALoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4LoRALoader": "WINT4 LoRA Loader"}
