"""
wint4_lora_common.py
────────────────────
Shared LoRA key normalization utilities for WINT4 LoRA nodes.
Used by both WINT4LoRALoader and WINT4LoRAStack.
"""


def _normalize_layer_path(path: str) -> str | None:
    """
    Normalize any LoRA layer path to standard diffusion_model.blocks.N.attn.wq format.

    Handles:
      PREFIX  — transformer.*, blocks.*, lora_unet_*, lycoris_*, lora_transformer_*
      SUFFIX  — .to_q→.wq  .to_k→.wk  .to_v→.wv  .to_out.0→.wo  .to_gate→.gate
      BLOCK   — .ff.→.mlp.
      SEP     — underscore → dot  (for lora_unet_ / lycoris_ / lora_transformer_)

    Returns None for paths that cannot be mapped (e.g. img_in, final_layer).
    """
    # ── Step 0: underscore formats → dot format ───────────────────
    und_prefix = None
    for pf in ["lora_transformer_", "lora_unet_", "lycoris_"]:
        if path.startswith(pf):
            und_prefix = pf
            break
    if und_prefix is not None:
        rest = path[len(und_prefix):]
        path = rest.replace("_", ".")

    # ── Step 1: normalize prefix → diffusion_model.blocks.N ────────
    if path.startswith("transformer."):
        rest = path[len("transformer."):]
        if rest.startswith("img_in") or rest.startswith("final_layer"):
            return None
        if rest.startswith("text_fusion.layerwise_blocks."):
            rest = rest[len("text_fusion.layerwise_blocks."):]
        elif rest.startswith("blocks."):
            rest = rest[len("blocks."):]
        else:
            return None
        path = f"diffusion_model.blocks.{rest}"
    elif path.startswith("diffusion_model."):
        pass
    elif path.startswith("blocks."):
        path = f"diffusion_model.{path}"
    else:
        return None

    # ── Step 2: normalize block type  .ff. → .mlp. ───────────────
    path = path.replace(".ff.", ".mlp.")

    # ── Step 3: normalize suffixes ────────────────────────────────
    path = path.replace(".to_q", ".wq")
    path = path.replace(".to_k", ".wk")
    path = path.replace(".to_v", ".wv")
    path = path.replace(".to_out.0", ".wo")
    path = path.replace(".to_out", ".wo")
    path = path.replace(".to_gate", ".gate")

    return path


def _auto_detect_format(sd: dict) -> str:
    """Detect LoRA key format: 'standard', 'bfl', or 'unknown'."""
    for key in sd:
        if "single_blocks" in key or "double_blocks" in key:
            return "bfl"
        if "diffusion_model.blocks" in key:
            return "standard"
    return "unknown"


def _convert_bfl_to_standard(sd: dict) -> dict:
    """Convert BFL-format LoRA keys to standard ComfyUI format.

    BFL: single_blocks.0.attn.qkv.lora_A.weight
      →  diffusion_model.blocks.0.attn.qkv.lora_up.weight

    lora_A = down projection, lora_B = up projection.
    """
    out = {}
    for key, tensor in sd.items():
        if "qkv.lora" in key or "proj.lora" in key or "ff.lora" in key:
            for prefix in ["double_blocks", "single_blocks"]:
                if key.startswith(prefix):
                    break
            else:
                out[key] = tensor
                continue
            rest = key[len(prefix) + 1:]
            parts = rest.split(".")
            block_num = parts[0]
            attn_type = parts[1] if len(parts) > 1 and "attn" in parts[1] else "attn"
            if "lora_B" in key:
                lora_type = "up"
            elif "lora_A" in key:
                lora_type = "down"
            elif "lora_up" in key:
                lora_type = "up"
            elif "lora_down" in key:
                lora_type = "down"
            else:
                out[key] = tensor
                continue
            stem = "qkv" if "qkv" in key else "proj"
            std_key = f"diffusion_model.blocks.{block_num}.{attn_type}.{stem}"
            out[f"{std_key}.lora_{lora_type}.weight"] = tensor
        else:
            out[key] = tensor
    return out
