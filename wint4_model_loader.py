"""
wint4_model_loader.py
─────────────────────
WINT4 Model Loader node for ComfyUI.

Loads an INT4-quantized (packed uint8) diffusion model using Int4XPUOps.
"""
import logging
import folder_paths
import comfy.sd
import comfy.model_detection

log = logging.getLogger("WINT4-Loader")

NODE_NAME = "WINT4 Model Loader"

# ── preserve original detect_unet_config for monkey-patch ──
_orig_detect_unet_config = comfy.model_detection.detect_unet_config


def _detect_with_int4_fallback(state_dict, key_prefix, metadata=None):
	"""Wrapper: native detection first; INT4 Wan fallback if that fails."""
	result = _orig_detect_unet_config(state_dict, key_prefix, metadata)
	if result is not None:
		return result

	keys = list(state_dict.keys())
	# Wan fingerprint: 5-D patch_embedding.weight  [dim, 16, 1, 2, 2]
	pe_key = "{}patch_embedding.weight".format(key_prefix)
	if pe_key not in keys:
		return None
	pe = state_dict[pe_key]
	if pe.ndim != 5 or pe.shape[1] != 16:
		return None

	ffn_key = "{}blocks.0.ffn.0.weight".format(key_prefix)
	if ffn_key not in keys:
		return None

	dim = int(pe.shape[0])
	ffn_dim = int(state_dict[ffn_key].shape[0])
	in_dim = int(pe.shape[1])
	num_layers = comfy.model_detection.count_blocks(
		keys, "{}blocks.".format(key_prefix) + "{}."
	)

	out_dim = 16
	dit_config = {
		"image_model": "wan2.1",
		"dim": dim,
		"out_dim": out_dim,
		"num_heads": dim // 128,
		"ffn_dim": ffn_dim,
		"num_layers": num_layers,
		"patch_size": (1, 2, 2),
		"freq_dim": 256,
		"window_size": (-1, -1),
		"qk_norm": True,
		"cross_attn_norm": True,
		"eps": 1e-6,
		"in_dim": in_dim,
	}

	# subtype: i2v vs t2v (mirrors original detection logic)
	if "{}img_emb.proj.0.bias".format(key_prefix) in keys:
		dit_config["model_type"] = "i2v"
	else:
		dit_config["model_type"] = "t2v"

	return dit_config


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

		# ── monkey-patch detect_unet_config with INT4 fallback ──
		comfy.model_detection.detect_unet_config = _detect_with_int4_fallback
		try:
			model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
		finally:
			comfy.model_detection.detect_unet_config = _orig_detect_unet_config

		# ── Mark model for LoRA reset on next load ───────────────
		object.__setattr__(model.model, '_lora_needs_reset', True)

		# ── Patch detach to clear _lora_entries before offload ──
		_orig_detach = model.detach
		def _detach_with_cleanup(unpatch_all=True):
			dm = model.model.diffusion_model
			for module in dm.modules():
				if hasattr(module, '_lora_entries'):
					object.__setattr__(module, '_lora_entries', {})
				bake_state = getattr(module, '_wint4_bake_state', None)
				if bake_state is not None and '_orig_weight' in bake_state:
					module.weight.data.copy_(bake_state['_orig_weight'])
				object.__setattr__(module, '_wint4_bake_state', None)
			return _orig_detach(unpatch_all)
		object.__setattr__(model, 'detach', _detach_with_cleanup)

		log.info(
			f"[WINT4 Loader] Loaded '{unet_name}' | type={model_type} "
			f"| INT4 VRAM savings active"
		)
		return (model,)


NODE_CLASS_MAPPINGS = {"WINT4ModelLoader": WINT4ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4ModelLoader": "WINT4 Model Loader"}
