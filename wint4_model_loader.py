"""
wint4_model_loader.py
─────────────────────
WINT4 Model Loader node for ComfyUI.

Loads an INT4-quantized (packed uint8) diffusion model using Int4XPUOps.
"""

import logging
import folder_paths
import comfy.sd

log = logging.getLogger("WINT4-Loader")

NODE_NAME = "WINT4 Model Loader"


class WINT4ModelLoader:

    NAME = NODE_NAME
    CATEGORY = "WINT4"

    @classmethod
    def INPUT_TYPES(cls):
        from .wint4_model_quantizer import MODEL_TYPES
        return {
            "required": {
                "unet_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "INT4 model produced by WINT4ModelQuantizer"},
                ),
                "model_type": (
                    MODEL_TYPES,
                    {
                        "default": "flux2",
                        "tooltip": "Must match the type used during quantization",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"

    def load_model(self, unet_name: str, model_type: str):
        from .wint4_xpu_ops import Int4XPUOps
        from .wint4_model_quantizer import _EXCLUSIONS

        Int4XPUOps.excluded_names = _EXCLUSIONS.get(model_type, [])
        Int4XPUOps._is_prequantized = False

        model_options = {"custom_operations": Int4XPUOps}

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if unet_path is None:
            raise FileNotFoundError(
                f"[WINT4 Loader] Model '{unet_name}' not found in diffusion_models."
            )

        log.info(
            f"[WINT4 Loader] Loading: {unet_name} (type={model_type})"
        )
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        log.info(
            f"[WINT4 Loader] Loaded '{unet_name}' | type={model_type} "
            f"| INT4 VRAM savings active"
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"WINT4ModelLoader": WINT4ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4ModelLoader": "WINT4 Model Loader"}
