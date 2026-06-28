"""
wint4_xpu_ops.py
────────────────
INT4 custom operations for Intel XPU (Arc A770).

Packed INT4 inference (uint8 storage, 2×4bit per byte):
  - weight: (out_f, in_f // 2) uint8, 偶数列低4位/奇数列高4位
  - weight_scale: (out_f, 1) float32
  - unpack → dequant → F.linear

When AIMDO DynamicVRAM is ON:  uses cast_bias_weight / uncast_bias_weight.
When AIMDO DynamicVRAM is OFF: local-variable device alignment,
  self.weight always stays on CPU, VRAM released after forward.

LoRA: NOT supported on INT4 packed layers (shape mismatch unavoidable).
      Excluded layers (first/last/norm) stay BF16 and LoRA works on those.
"""

import os
import json
import logging

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

log = logging.getLogger("WINT4-XPU")

# ═══════════════════════════════════════════════════════════════════════════════
# Environment setup
# ═══════════════════════════════════════════════════════════════════════════════

_TRITON_AVAILABLE = False


def _try_add_dll_search_paths():
    candidates = []
    try:
        import folder_paths
        base = folder_paths.base_path
        if base:
            candidates.append(os.path.join(base, ".ext", "Library", "bin"))
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rel = os.path.normpath(os.path.join(here, "..", "..", "..", ".ext", "Library", "bin"))
        candidates.append(rel)
    except Exception:
        pass
    for p in candidates:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass


def _find_oneapi_2025():
    base = r"C:\Program Files (x86)\Intel\oneAPI\compiler"
    if not os.path.isdir(base):
        return None
    versions = []
    for e in os.listdir(base):
        full = os.path.join(base, e)
        if os.path.isdir(full) and e.startswith("2025."):
            if os.path.isfile(os.path.join(full, "bin", "icpx.exe")):
                versions.append(e)
    versions.sort(reverse=True)
    return os.path.join(base, versions[0]) if versions else None


def _patch_compilation_helper():
    global _TRITON_AVAILABLE
    try:
        from triton.backends.intel.driver import COMPILATION_HELPER
    except ImportError:
        return
    oneapi_dir = _find_oneapi_2025()
    if oneapi_dir is None:
        return
    level_zero_dir = r"C:\Program Files\LevelZeroSDK\1.28.2"
    triton_dir = os.path.dirname(triton.__file__)
    triton_inc = os.path.join(triton_dir, "backends", "intel", "include")
    triton_lib = os.path.join(triton_dir, "backends", "intel", "lib")
    COMPILATION_HELPER.include_dir = [
        triton_inc,
        os.path.join(oneapi_dir, "include"),
        os.path.join(oneapi_dir, "include", "sycl"),
        os.path.join(level_zero_dir, "include"),
    ]
    COMPILATION_HELPER.library_dir = [
        triton_lib,
        os.path.join(oneapi_dir, "lib"),
        os.path.join(level_zero_dir, "lib"),
    ]
    icpx_bin = os.path.join(oneapi_dir, "bin")
    if os.path.isdir(icpx_bin) and icpx_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = icpx_bin + os.pathsep + os.environ.get("PATH", "")
    log.info(f"WINT4-XPU: Triton compilation locked to {oneapi_dir}")


_try_add_dll_search_paths()
_patch_compilation_helper()

try:
    _TRITON_AVAILABLE = True
    log.info("WINT4-XPU: Triton XPU available")
except ImportError:
    log.info("WINT4-XPU: Triton not available")


def _aimdo_active() -> bool:
    """Check if AIMDO DynamicVRAM is currently enabled."""
    try:
        from comfy_aimdo import control as _ctrl
        return _ctrl.is_dynamic_vram_enabled()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI custom operations
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS = True
except ImportError:
    _COMFY_OPS = False

if _COMFY_OPS:

    class Int4XPUOps(manual_cast):
        excluded_names: list = []
        _is_prequantized: bool = False

        class Linear(manual_cast.Linear):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_buffer("weight_scale", None)
                self._is_quantized = False
                self._use_quarot = False
                self._group_size = 128
                self._hadamard_H = None
                self.compute_dtype = torch.float16

            def _load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            ):
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                bias_key = prefix + "bias"
                meta_key = prefix + "comfy_quant"
                input_scale_key = prefix + "input_scale"

                weight_tensor = state_dict.pop(weight_key, None)
                weight_scale = state_dict.pop(scale_key, None)
                meta_raw = state_dict.pop(meta_key, None)
                state_dict.pop(input_scale_key, None)

                if weight_tensor is None:
                    missing_keys.append(weight_key)
                    self._is_quantized = False
                elif weight_tensor.dtype == torch.uint8 and weight_scale is not None:
                    # INT4 packed: (out_f, in_f // 2) uint8
                    Int4XPUOps._is_prequantized = True
                    self._is_quantized = True
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    self.register_buffer("weight_scale", weight_scale.float())

                    if meta_raw is not None:
                        try:
                            meta = json.loads(bytes(meta_raw.tolist()).decode("utf-8"))
                            is_rotated = meta.get("quarot", False) or meta.get("convrot", False)
                            if is_rotated:
                                self._use_quarot = True
                                gs = meta.get("group_size", meta.get("convrot_groupsize", 128))
                                self._group_size = gs
                                from .wint8_quarot import build_hadamard
                                self._hadamard_H = build_hadamard(
                                    gs, device="cpu", dtype=torch.float32
                                )
                        except Exception:
                            pass

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    self._is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)

                bias_tensor = state_dict.pop(bias_key, None)
                self.bias = (
                    torch.nn.Parameter(bias_tensor, requires_grad=False)
                    if bias_tensor is not None else None
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                need_cast = (
                    self.comfy_cast_weights
                    or len(getattr(self, 'weight_function', [])) > 0
                    or len(getattr(self, 'bias_function', [])) > 0
                )

                if not self._is_quantized:
                    if need_cast:
                        weight, bias, offload_stream = cast_bias_weight(
                            self, x, offloadable=True,
                        )
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    return F.linear(x, self.weight, self.bias)

                if _aimdo_active():
                    return self._forward_aimdo(x, need_cast)
                else:
                    return self._forward_simple(x, need_cast)

            def _forward_aimdo(self, x, need_cast):
                weight, bias, offload_stream = cast_bias_weight(
                    self, x, offloadable=True,
                )
                result = self._compute(x, weight, bias, need_cast)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return result

            def _forward_simple(self, x, need_cast):
                w = self.weight.to(x.device, non_blocking=True, dtype=torch.uint8)
                b = self.bias.to(device=x.device) if self.bias is not None else None
                result = self._compute(x, w, b, need_cast)
                if x.device.type == 'xpu':
                    torch.xpu.empty_cache()
                return result

            def _compute(self, x, weight, bias, need_cast):
                w_scale = self.weight_scale
                if w_scale is not None and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                x2 = x.reshape(-1, x.shape[-1])
                comp_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16

                weight_u8 = weight.to(torch.uint8, non_blocking=True)
                out_f = weight_u8.shape[0]
                in_f = weight_u8.shape[1] * 2

                w_low = (weight_u8 & 0x0F).to(torch.float16)
                w_high = ((weight_u8 >> 4) & 0x0F).to(torch.float16)
                w_unpacked = torch.cat([w_low.unsqueeze(-1), w_high.unsqueeze(-1)], dim=-1)
                del w_low, w_high
                w_unpacked = w_unpacked.reshape(out_f, in_f).sub_(8.0)

                if self._use_quarot and self._hadamard_H is not None:
                    try:
                        from .wint8_quarot import rotate_activation
                        x2 = rotate_activation(x2, self._hadamard_H, self._group_size)
                    except Exception:
                        pass

                if w_scale.ndim >= 1 and w_scale.shape[0] > 1:
                    w_dq = w_unpacked.mul(w_scale.view(-1, 1)).to(comp_dtype)
                else:
                    w_dq = w_unpacked.mul(w_scale).to(comp_dtype)
                del w_unpacked

                b_dq = bias.to(device=x.device, dtype=comp_dtype) if bias is not None else None

                if need_cast:
                    for fn in getattr(self, 'weight_function', []):
                        w_dq = fn(w_dq)
                    for fn in getattr(self, 'bias_function', []):
                        if b_dq is not None:
                            b_dq = fn(b_dq)

                out = F.linear(x2.to(comp_dtype), w_dq, b_dq)
                del w_dq

                return out.reshape(*x.shape[:-1], out.shape[-1])

        class GroupNorm(manual_cast.GroupNorm):
            pass
        class LayerNorm(manual_cast.LayerNorm):
            pass
        class Conv2d(manual_cast.Conv2d):
            pass
        class Conv3d(manual_cast.Conv3d):
            pass
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass
        class Embedding(manual_cast.Embedding):
            pass

        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            raise ValueError(f"Int4XPUOps: unsupported conv dims: {dims}")

else:
    class Int4XPUOps:
        pass
