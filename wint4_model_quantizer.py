"""
wint4_model_quantizer.py
────────────────────────
Standalone INT4 model quantizer node for ComfyUI.

Per-row INT4 quantization with manual uint8 packing (2×4bit per byte).
Supports input: BF16 / FP16 / FP8 / INT8 → output: INT4 packed uint8.
Same exclusion list & QuaRot support as INT8.

Features:
  - Odd in_features auto-padded to even before packing
  - Auto-creates output subdirectories (e.g. int4/my_model)
  - Converts residual FP8 tensors to FP16 for XPU compatibility
  - Multi-device: XPU / CUDA / MPS / CPU
  - Boogu: auto-uses group_size=32 for full QuaRot coverage
  - Preserves source model metadata (config) for LTX/Wan version detection
"""

import os
import json
import logging

import torch
import folder_paths
import comfy.utils
from safetensors import safe_open

from .wint8_quarot import build_hadamard, rotate_weight

log = logging.getLogger("WINT4-Quantizer")

# ── Exclusion lists (shared with INT8) ──────────────────────────────────────

_EXCLUSIONS = {
    "flux2": [
        "img_in", "time_in", "guidance_in", "txt_in", "final_layer",
        "double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation",
    ],
    "z-image": [
        "cap_embedder", "t_embedder", "x_embedder", "cap_pad_token",
        "context_refiner", "final_layer", "noise_refiner", "adaLN",
        "x_pad_token", "layers.0.",
        "cap_embedder.0", "attention_norm1", "attention_norm2",
        "ffn_norm1", "ffn_norm2", "k_norm", "q_norm",
    ],
    "chroma": [
        "distilled_guidance_layer", "final_layer", "img_in", "txt_in",
        "nerf_image_embedder", "nerf_blocks", "nerf_final_layer_conv",
        "__x0__",
    ],
    "wan": [
        "patch_embedding", "text_embedding", "time_embedding",
        "time_projection", "head", "img_emb", "motion_encoder",
        "modulation", "norm_q", "norm_k", "norm3",
    ],
    "ltx2": [
        "adaln_single", "audio_adaln_single", "audio_caption_projection",
        "audio_patchify_proj", "audio_proj_out", "audio_scale_shift_table",
        "av_ca_a2v_gate_adaln_single", "av_ca_audio_scale_shift_adaln_single",
        "av_ca_v2a_gate_adaln_single", "av_ca_video_scale_shift_adaln_single",
        "caption_projection", "patchify_proj", "proj_out", "scale_shift_table",
        "learnable_registers", "q_norm", "k_norm",
    ],
    "qwen": [
        "time_text_embed", "img_in", "norm_out", "proj_out", "txt_in",
        "norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm",
        "transformer_blocks.0.img_mod.1",
    ],
    "ernie": [
        "time", "x_embedder", "adaLN", "final", "text_proj",
        "norm", "layers.0.", "layers.35",
    ],
    "hidream": [
        "patch_embedding", "time_text_embed", "norm_out", "proj_out",
    ],
    "boogu": [
        "embed", "refine", "norm_out",
    ],
    "krea2": [
        "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp",
    ],
    "ideogram4": [
        "embed_image_indicator", "t_embedding", "proj",
    ],
    "auto": [],
}

MODEL_TYPES = list(_EXCLUSIONS.keys())

# ── Device helpers ────────────────────────────────────────────────────────────

def _get_available_devices() -> list[str]:
    choices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        choices.append("xpu")
    if torch.cuda.is_available():
        choices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        choices.append("mps")
    return choices


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Exclusion helpers ─────────────────────────────────────────────────────────

def _is_excluded(key: str, model_type: str) -> bool:
    for pattern in _EXCLUSIONS.get(model_type, []):
        if pattern in key:
            return True
    return False


def _should_quantize(key: str, tensor: torch.Tensor, model_type: str) -> bool:
    if tensor.ndim != 2:
        return False
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                             torch.float8_e4m3fn, torch.float8_e5m2,
                             torch.int8):
        return False
    if _is_excluded(key, model_type):
        return False
    return True


class WINT4ModelQuantizer:

    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("diffusion_models")
        if not files:
            files = ["none"]
        devices = _get_available_devices()
        device_default = "xpu" if "xpu" in devices else ("cuda" if "cuda" in devices else "cpu")
        return {
            "required": {
                "model_name": (files, {"tooltip": "Source model (BF16/FP16/FP8/INT8)"}),
                "model_type": (MODEL_TYPES, {"default": "flux2"}),
                "enable_quarot": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Hadamard rotation (QuaRot/ConvRot) for better quality.",
                }),
                "group_size": ("INT", {
                    "default": 128, "min": 64, "max": 256, "step": 64,
                    "tooltip": "QuaRot group size. Boogu auto-overrides to 32.",
                }),
                "device": (devices, {
                    "default": device_default,
                    "tooltip": "Device used during quantization.  Supports XPU / CUDA / MPS / CPU.",
                }),
                "output_filename": ("STRING", {
                    "default": "model_int4",
                    "tooltip": "Saved to ComfyUI/output/.  Supports subdirectories (e.g. int4/my_model).",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "quantize"
    OUTPUT_NODE = True
    CATEGORY = "WINT4"
    DESCRIPTION = "Quantize a diffusion model to per-row INT4 (packed uint8). Supports BF16/FP16/FP8/INT8 input."

    def quantize(
        self,
        model_name: str,
        model_type: str,
        enable_quarot: bool,
        group_size: int,
        device: str,
        output_filename: str,
    ):
        src_path = folder_paths.get_full_path("diffusion_models", model_name)
        if src_path is None:
            raise FileNotFoundError(f"Model '{model_name}' not found.")

        output_dir = folder_paths.get_output_directory()
        dst_path = os.path.join(output_dir, f"{output_filename}.safetensors")

        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)

        dev = _resolve_device(device)
        log.info(f"[WINT4 Quantizer] Device: {dev}  (per-row INT4)")

        sd = comfy.utils.load_torch_file(src_path, safe_load=True)
        log.info(f"[WINT4 Quantizer] Loaded {len(sd)} keys.")

        # ── Preserve source metadata (config for LTX/Wan version detection) ──
        src_metadata = {}
        try:
            with safe_open(src_path, framework="pt") as f:
                src_metadata = f.metadata() or {}
        except Exception:
            pass

        # ── Boogu: auto-override group_size to 32 for full coverage ──
        H = None
        quarot_applied = False
        if enable_quarot:
            actual_gs = group_size
            if model_type == "boogu":
                actual_gs = 32
                log.info(
                    f"[WINT4 Quantizer] Boogu detected — group_size overridden "
                    f"to {actual_gs} for full layer coverage"
                )
            H = build_hadamard(actual_gs, device=str(dev), dtype=torch.float32)
            log.info(f"[WINT4 Quantizer] QuaRot enabled, group_size={actual_gs}")

        quantized_count = 0
        excluded_count = 0
        total_before_bytes = 0
        total_after_bytes = 0

        for key in list(sd.keys()):
            tensor = sd[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            if not _should_quantize(key, tensor, model_type):
                if tensor.ndim == 2 and _is_excluded(key, model_type):
                    excluded_count += 1
                continue

            if tensor.dtype == torch.int8:
                base_key = key.rsplit(".weight", 1)[0]
                scale_key = f"{base_key}.weight_scale"
                if scale_key in sd:
                    w_scale_src = sd[scale_key].float().to(dev)
                    if w_scale_src.ndim == 2:
                        w = (tensor.float().to(dev) * w_scale_src)
                    else:
                        w = (tensor.float().to(dev) * w_scale_src.view(-1, 1))
                else:
                    log.warning(f"[WINT4 Quantizer] INT8 weight without scale: {key}, using raw float")
                    w = tensor.float().to(dev)
            else:
                w = tensor.float().to(dev)

            layer_quarot = False
            if H is not None and w.shape[1] % actual_gs == 0:
                try:
                    w = rotate_weight(w, H, group_size=actual_gs)
                    layer_quarot = True
                    quarot_applied = True
                except ValueError:
                    pass

            amax = w.abs().amax(dim=1, keepdim=True)
            scale = (amax / 7.0).clamp(min=1e-8)
            q = (w / scale).round().clamp(-8, 7).add(8).to(torch.uint8)

            pad_dim = q.shape[1] % 2
            if pad_dim:
                q = torch.cat([q, torch.zeros(q.shape[0], 1, dtype=torch.uint8, device=q.device)], dim=1)

            q_packed = (q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)

            base = key.rsplit(".weight", 1)[0]

            sd[key] = q_packed.cpu()
            sd[f"{base}.weight_scale"] = scale.cpu()
            sd[f"{base}.comfy_quant"] = _make_comfy_quant(
                quarot=layer_quarot,
                group_size=actual_gs if layer_quarot else None,
            )
            sd[f"{base}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)

            total_before_bytes += tensor.numel() * tensor.element_size()
            total_after_bytes += q_packed.numel() * 1 + scale.numel() * 4
            quantized_count += 1
            del w, q, q_packed, scale

        if dev.type in ("xpu", "cuda"):
            try:
                (torch.xpu if dev.type == "xpu" else torch.cuda).empty_cache()
            except Exception:
                pass

        for k in list(sd.keys()):
            v = sd[k]
            if isinstance(v, torch.Tensor) and v.dtype in (
                torch.float8_e4m3fn, torch.float8_e5m2,
            ):
                if k.endswith('.weight'):
                    base = k.rsplit('.weight', 1)[0]
                    if f"{base}.weight_scale" in sd:
                        del sd[k]
                        continue
                sd[k] = v.to(torch.float16)

        sd["int4_quantized"] = torch.tensor(1, dtype=torch.uint8)
        sd["int4_model_type"] = _str_to_uint8_tensor(model_type)

        # ── Save with source metadata preserved ──────────────────────
        save_kwargs = {}
        if src_metadata:
            save_kwargs["metadata"] = src_metadata

        log.info(f"[WINT4 Quantizer] Writing {dst_path} ...")
        comfy.utils.save_torch_file(sd, dst_path, **save_kwargs)

        mb_before = total_before_bytes / (1024 * 1024)
        mb_after = total_after_bytes / (1024 * 1024)
        log.info(
            f"\n{'='*60}\n"
            f"  WINT4 Quantization Complete (per-row INT4)\n"
            f"  Model: {model_name} | Type: {model_type}\n"
            f"  Device: {dev} | QuaRot: {quarot_applied}\n"
            f"  Quantized: {quantized_count} | Excluded: {excluded_count}\n"
            f"  Weight: {mb_before:.1f} MB → {mb_after:.1f} MB "
            f"({100*mb_after/max(mb_before,1):.0f}%)\n"
            f"  Output: {dst_path}\n"
            f"  {'='*60}"
        )
        if src_metadata:
            has_config = "config" in src_metadata
            log.info(f"[WINT4 Quantizer] Source metadata preserved ({'with config' if has_config else 'no config'})")
        return ()


def _make_comfy_quant(quarot: bool = False, group_size: int | None = None) -> torch.Tensor:
    payload = {"format": "int4_tensorwise", "per_row": True}
    if quarot and group_size:
        payload["convrot"] = True
        payload["convrot_groupsize"] = group_size
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def _str_to_uint8_tensor(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)


NODE_CLASS_MAPPINGS = {"WINT4ModelQuantizer": WINT4ModelQuantizer}
NODE_DISPLAY_NAME_MAPPINGS = {"WINT4ModelQuantizer": "WINT4 Model Quantizer"}
