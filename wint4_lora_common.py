"""
wint4_lora_common.py
────────────────────
Shared LoRA key normalization utilities for WINT4 LoRA nodes.
Used by both WINT4LoRALoader and WINT4LoRAStack.

_normalize_layer_path maps ANY LoRA key OR model internal module name
to a single canonical form:  diffusion_model.blocks.N.attn.wq

This means model internal names and LoRA keys are normalized to the
SAME space, and matching is exact string comparison after normalization.
"""


def _normalize_layer_path(path: str) -> str | None:
    # ── Step 0: underscore formats → dot format ───────────────────
    stripped_prefix = None
    for pf in ["lora_transformer_", "lora_unet_", "lycoris_"]:
        if path.startswith(pf):
            path = path[len(pf):].replace("_", ".")
            stripped_prefix = pf
            break

    if stripped_prefix is None:
        if path.startswith("transformer."):
            path = path[len("transformer."):]
            stripped_prefix = "transformer."
        elif path.startswith("diffusion_model."):
            path = path[len("diffusion_model."):]
            stripped_prefix = "diffusion_model."

    # ── Step 1: excluded layers ──────────────────────────────────
    if path.startswith("img_in") or path.startswith("final_layer"):
        return None

    # ── Step 2: sub-structure → blocks ───────────────────────────
    if path.startswith("text_fusion.layerwise_blocks."):
        path = "blocks." + path[len("text_fusion.layerwise_blocks."):]

    for old, new in [
        ("layers.", "blocks."),
        ("joint_blocks.", "blocks."),
        ("transformer_blocks.", "blocks."),
        ("double_blocks.", "blocks."),
        ("single_blocks.", "blocks."),
    ]:
        if path.startswith(old):
            path = new + path[len(old):]
            break

    # ── Step 3: block type normalization ─────────────────────────
    path = path.replace(".ff.", ".mlp.")
    path = path.replace(".feed_forward.", ".mlp.")

    # ── Step 4: substructure within attention ───────────────────
    path = path.replace(".img_attn.", ".attn.")
    path = path.replace(".txt_attn.", ".attn.")
    path = path.replace(".attention.", ".attn.")

    # ── Step 5: suffix normalization ────────────────────────────
    path = path.replace(".to_q", ".wq")
    path = path.replace(".to_k", ".wk")
    path = path.replace(".to_v", ".wv")
    path = path.replace(".to_out.0", ".wo")
    path = path.replace(".to_out", ".wo")
    path = path.replace(".to_gate", ".gate")

    path = path.replace(".q_proj", ".wq")
    path = path.replace(".k_proj", ".wk")
    path = path.replace(".v_proj", ".wv")
    path = path.replace(".out_proj", ".wo")

    path = path.replace(".self_attn.q", ".attn.wq")
    path = path.replace(".self_attn.k", ".attn.wk")
    path = path.replace(".self_attn.v", ".attn.wv")
    path = path.replace(".self_attn.o", ".attn.wo")

    # Z-image: attention output is named "out" (not "wo")
    path = path.replace(".attn.out", ".attn.wo")

    return f"diffusion_model.{path}"


def _auto_detect_format(sd: dict) -> str:
    for key in sd:
        if "single_blocks" in key or "double_blocks" in key:
            return "bfl"
        if "diffusion_model.blocks" in key or "diffusion_model.layers" in key:
            return "standard"
    return "unknown"


def _convert_bfl_to_standard(sd: dict) -> dict:
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
