"""
WINT4 Model Quantizer + Loader
───────────────────────────────
Standalone INT4 per-row model quantization & loading for ComfyUI.
Packed uint8 storage (2×4bit per byte).

Two nodes:
  WINT4ModelQuantizer  → offline per-row INT4 quantization
                         (BF16/FP16/FP8/INT8 → INT4 packed uint8)
  WINT4ModelLoader     → load INT4 models with VRAM savings

Supports: BF16, FP16, FP8, INT8 input models
QuaRot: optional Hadamard rotation for better quality
AIMDO: compatible (auto-detect, dual path)
LoRA: NOT supported on INT4 packed layers (excluded layers work)

Pure PyTorch — no CUDA, no external dependencies beyond ComfyUI itself.
Works on CPU / CUDA / XPU / any PyTorch backend.
"""

from .wint4_model_quantizer import (
    NODE_CLASS_MAPPINGS as _QUANT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _QUANT_DISPLAY,
)

from .wint4_model_loader import (
    NODE_CLASS_MAPPINGS as _LOAD_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LOAD_DISPLAY,
)

NODE_CLASS_MAPPINGS = {
    **_QUANT_MAPPINGS,
    **_LOAD_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_QUANT_DISPLAY,
    **_LOAD_DISPLAY,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
