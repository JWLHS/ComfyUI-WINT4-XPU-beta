"""
WINT4 Model Quantizer + Loader + LoRA
─────────────────────────────────────
Standalone INT4 per-row model quantization, loading & LoRA for ComfyUI.

Nodes:
  WINT4ModelQuantizer  → UNet BF16/FP16/FP8/INT8 → INT4 packed uint8
  WINT4ModelLoader     → load INT4 UNet
  WINT4LoRALoader      → single LoRA for INT4 UNet
  WINT4LoRAStack       → multi-LoRA for INT4 UNet
"""

from .wint4_model_quantizer import NODE_CLASS_MAPPINGS as _Q, NODE_DISPLAY_NAME_MAPPINGS as _QD
from .wint4_model_loader import NODE_CLASS_MAPPINGS as _L, NODE_DISPLAY_NAME_MAPPINGS as _LD
from .wint4_lora_loader import NODE_CLASS_MAPPINGS as _LR, NODE_DISPLAY_NAME_MAPPINGS as _LRD
from .wint4_lora_stack import NODE_CLASS_MAPPINGS as _LS, NODE_DISPLAY_NAME_MAPPINGS as _LSD

NODE_CLASS_MAPPINGS = {**_Q, **_L, **_LR, **_LS}
NODE_DISPLAY_NAME_MAPPINGS = {**_QD, **_LD, **_LRD, **_LSD}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
