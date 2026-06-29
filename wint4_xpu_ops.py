"""
wint4_xpu_ops.py
────────────────
INT4 custom operations for Intel XPU (Arc A770).

Packed INT4 inference (uint8 storage, 2×4bit per byte):
  - weight: (out_f, in_f // 2) uint8
  - weight_scale: (out_f, 1) float32
  - unpack → dequant → F.linear

LoRA: via _lora_entries dict {lora_name: [(A,B,multiplier[,start,end]), ...]}.
A = down projection (rank, in_f), B = up projection (out_f, rank).
Stored on XPU.  Forward computes delta = B @ A on-the-fly.
"""
import json
import logging
import torch
import torch.nn.functional as F

log = logging.getLogger("WINT4-XPU")

def _aimdo_active() -> bool:
    try:
        from comfy_aimdo import control as _ctrl
        return _ctrl.is_dynamic_vram_enabled()
    except Exception:
        return False

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
                                self._hadamard_H = build_hadamard(gs, device="cpu", dtype=torch.float32)
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
                        weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                        out = F.linear(x, weight, bias)
                        uncast_bias_weight(self, weight, bias, offload_stream)
                        return out
                    return F.linear(x, self.weight, self.bias)
                if _aimdo_active():
                    return self._forward_aimdo(x, need_cast)
                else:
                    return self._forward_simple(x, need_cast)

            def _forward_aimdo(self, x, need_cast):
                weight, bias, offload_stream = cast_bias_weight(self, x, offloadable=True)
                result = self._compute(x, weight, bias, need_cast)
                uncast_bias_weight(self, weight, bias, offload_stream)
                if x.device.type == 'xpu':
                    torch.xpu.empty_cache()
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

                # ── WINT4 LoRA: delta = B @ A on XPU ──────────────
                # _lora_entries = dict { lora_name: [(A,B,mult,...), ...] }
                lora_entries = getattr(self, '_lora_entries', None)
                if lora_entries is not None:
                    for entries_list in lora_entries.values():
                        for entry in entries_list:
                            A, B, multiplier = entry[:3]
                            sl_start = entry[3] if len(entry) > 3 else None
                            sl_end   = entry[4] if len(entry) > 4 else None

                            if A.shape[1] != w_dq.shape[1]:
                                continue

                            A_dev = A.to(dtype=comp_dtype) if A.dtype != comp_dtype else A
                            B_dev = B.to(dtype=comp_dtype) if B.dtype != comp_dtype else B
                            if A_dev.device != w_dq.device:
                                A_dev = A_dev.to(device=w_dq.device)
                            if B_dev.device != w_dq.device:
                                B_dev = B_dev.to(device=w_dq.device)

                            delta = B_dev @ A_dev
                            delta.mul_(multiplier)

                            if sl_start is not None:
                                w_dq[sl_start:sl_end, :].add_(delta)
                            else:
                                w_dq.add_(delta)

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
