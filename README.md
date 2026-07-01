

---

## `README.md`

```markdown
# ComfyUI-WINT4-XPU-beta

> **INT4 per-row 模型量化与 LoRA 推理插件 — 专为 Intel Arc A770 16GB 优化**

将扩散模型量化为 **per-row INT4 (packed uint8)**，显存节省 75%，支持多 LoRA 叠加（含 LoKr 和 ICLoRA bake-in）。

---

## 功能概览

| 能力 | 状态 |
|------|:--:|
| BF16/FP16/FP8/INT8 → INT4 量化 | ✅ |
| QuaRot (Hadamard 旋转) 质量提升 | ✅ **INT4 必须开启** |
| INT4 UNet 推理 (AIMDO 双路径) | ✅ |
| 单 LoRA 加载 | ✅ |
| 多 LoRA 叠加 (串联 / Stack) | ✅ |
| LyCORIS LoKr 格式 | ✅ |
| ICLoRA / adaLN LoRA (bake-in) | ✅ (v5.1) |
| QKV 融合模型 LoRA | ✅ |
| 7+ 种 LoRA key 格式自动适配 | ✅ |
| 源模型 metadata 保留 | ✅ (v5) |
| Wan 模型检测 fallback | ✅ (v5) |
| LTX2.3 完整支持 | ✅ (v5) |
| Boogu group_size=32 自动适配 | ✅ (v5.1) |
| TE 量化 | ❌ 已放弃 |

### 性能数据 (Krea2)

| 指标 | BF16 | INT8 | INT4 |
|------|:--:|:--:|:--:|
| UNet 存储 | 24 GB | 12 GB | **6 GB** |
| 推理显存 | ~24 GB | ~16–17 GB | **~8–9 GB** |
| A770 16GB 裸跑 | ❌ | ❌ | ✅ |

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta.git
```

依赖同 INT8 插件，ComfyUI 启动时自动安装。

---

## 节点

| 节点 | 功能 |
|------|------|
| `WINT4ModelQuantizer` | 量化 UNet → INT4 packed uint8 + QuaRot |
| `WINT4ModelLoader` | 加载 INT4 UNet |
| `WINT4LoRALoader` | 单 LoRA，支持链式串联 |
| `WINT4LoRAStack` | 多 LoRA 一次叠加 (最多 5 个) |

---

## 快速开始

```
1. 量化:
   WINT4ModelQuantizer
   ├─ model_name      = 你的 BF16/FP16 模型 (推荐 BF16)
   ├─ model_type      = krea2 / wan / ltx2 / flux2 / ...
   ├─ enable_quarot   = True  ← INT4 强烈建议开启
   ├─ group_size      = 64 (Boogu 自动用 32)
   ├─ device          = xpu
   └─ output_filename = my_model_int4

2. 推理 (无 LoRA):
   WINT4ModelLoader → MODEL → KSampler → VAE Decode

3. 推理 (单 LoRA):
   WINT4ModelLoader → WINT4LoRALoader → KSampler

4. 推理 (多 LoRA 串联):
   WINT4ModelLoader → WINT4LoRALoader (A) → WINT4LoRALoader (B) → ...

5. 推理 (多 LoRA Stack):
   WINT4ModelLoader → WINT4LoRAStack (最多 5 槽) → KSampler
```

LoRA 强度建议 **1.5–2.0×** (INT4 精度限制)。ICLoRA / adaLN LoRA 自动通过 bake-in 生效，无需额外配置。

---

## 支持的 LoRA 格式

| # | 格式 | 示例 |
|:-:|------|------|
| ① | Kohya 标准 | `diffusion_model.blocks.0.attn.wq.lora_B.weight` |
| ② | diffusers | `transformer.blocks.0.attn.to_q.lora_B.weight` |
| ③ | SimpleTuner lycoris | `lycoris_blocks_0_attn_wq.lora_down.weight` |
| ④ | bare | `blocks.0.attn.wq.lora_B.weight` |
| ⑤ | onetrainer | `transformer.text_fusion.layerwise_blocks.0...` |
| ⑥ | legacy ComfyUI | `lora_unet_blocks_0_attn_wq.lora_down.weight` |
| ⑦ | onetrainer alt | `lora_transformer_blocks_0_attn_wq.lora_down.weight` |
| ⑧ | BFL (Flux) | `single_blocks.0.attn.qkv.lora_A.weight` → 自动转换 |
| ⑨ | LyCORIS LoKr | `diffusion_model.blocks.0.attn.wq.lokr_w1` |
| ⑩ | LTX ICLoRA | `diffusion_model.adaln_single.*.lora_A.weight` → bake-in |

---

## 支持的模型

| model_type | 模型系列 | 验证状态 |
|:---|------|:--:|
| `flux2` | Flux2-Klein 系列 | ✅ |
| `qwen` | Qwen-Rapid / Qwen-EDIT / Qwen-AIO | ✅ |
| `z-image` | Z-Image 系列 | ✅ |
| `wan` | Wan 2.1 / 2.2 / SCAIL2 / WANanimate / Bernini / WANremix | ✅ |
| `ltx2` | LTX 2.3 (10EROSV1 等) | ✅ |
| `krea2` | Krea2 Turbo / Raw | ✅ |
| `boogu` | Boogu Base / Edit / Turbo (auto gs=32) | ✅ |
| `hidream` | HiDream 系列 | ✅ |
| `ernie` | ERNIE 系列 | ✅ |
| `ideogram4` | Ideogram 4 系列 | ✅ |
| `chroma` | Chroma/Distillation | ✅ |
| `auto` | 通用 (不排除任何层) | ⚠️ 谨慎使用 |

---

## FP8 源模型注意

FP8 源可以量化，但 **二次量化 (FP8→INT4) 精度损失较大**，推荐使用 BF16/FP16 源。如必须用 FP8 源，插件会自动转换残留 FP8 张量为 FP16 以保证 XPU 兼容。

---

## 双模型 / 多次推理

- 双采 (两个 INT4 UNet) — detach 时自动清理 `_lora_entries` 和 bake-in 权重
- 多次 prompt — `_lora_needs_reset` 标记自动恢复原始权重
- LoRA 热切换 — prune 机制清理过期条目

---

## v5 新增 (2026-07)

| 功能 | 说明 |
|------|------|
| **LTX2.3 完整支持** | 排除列表补全 + metadata 保留 |
| **Wan 模型检测 fallback** | `head.modulation` 缺失时从权重 shape 反推 config |
| **FP8 清理修复** | 仅删除已被量化的 FP8 权重，其余转 FP16 |
| **motion_encoder 保护** | Wan 排除列表加 `motion_encoder`，kernel 后缀回退 |
| **ICLoRA bake-in** | 非量化层 (adaln_single 等) 的 LoRA 直接融合到 weight |
| **Boogu group_size=32** | 自动检测并覆盖，达到 100% QuaRot 覆盖率 |
| **源 metadata 保留** | 保存时透传原始 `config`，ComfyUI 正确识别模型版本 |

---

## Bug 修复记录 (33 项, v1→v5)

| # | Bug | 修复 | 版本 |
|---|------|------|:--:|
| 28 | 多 LoRA 链式只有最后一个生效 | 删 `_lora_needs_reset=True`；prune 机制 | v4 |
| 29 | LoKr 格式匹配 0 层 | 解析 lokr_w1/w2；动态 Kronecker 展开 | v4 |
| 30 | LoKr 预计算 delta 撑爆显存 | 改动态展开 | v4 |
| 31 | 双采第一个 UNet 不卸载 | detach 清空 `_lora_entries` | v4 |
| 32 | LoKr delta shape 不匹配 | `_compute` shape guard | v4 |
| 33 | MPS 设备不支持 | 加 MPS 检测 | v4 |
| 34 | FP8 清理误删排除层 | 改为按 weight_scale 判断 + 转 FP16 | v5 |
| 35 | Wan 模型检测失败 (head 缺失) | 5D patch_embedding fingerprint fallback | v5 |
| 36 | motion_encoder 被量化 | wan 排除列表加 `motion_encoder` + kernel 回退 | v5 |
| 37 | LTX2.3 加载 shape mismatch | 排除列表补全 + metadata 保留 | v5 |
| 38 | Boogu QuaRot 仅 27% 覆盖 | 自动 group_size=32 | v5.1 |
| 39 | ICLoRA adaLN 部分被跳过 | bake-in 到非量化层 | v5.1 |

---

## 文件清单

```
ComfyUI-WINT4-XPU-beta/
├── __init__.py                  # 节点注册
├── wint4_model_quantizer.py     # 量化器 (含 metadata 保留)
├── wint4_model_loader.py        # 加载器 (含 Wan fallback)
├── wint4_xpu_ops.py             # 推理 ops (含 kernel 回退)
├── wint4_lora_loader.py         # 单 LoRA (含 bake-in)
├── wint4_lora_stack.py          # 多 LoRA Stack (含 bake-in)
├── wint4_lora_common.py         # key 归一化 + BFL 转换
├── wint8_quarot.py              # Hadamard 旋转
├── check_int4.py                # 模型诊断脚本
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 相关链接

- **本插件**: https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta
- **INT8 插件**: https://github.com/JWLHS/ComfyUI-WINT8-XPU
- **convert-to-quant**: https://github.com/newgrit1004/convert-to-quant

---

## 鸣谢

本插件全部由 **DeepSeek V4 Pro** 完成。

参考项目：
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-WINT8-XPU](https://github.com/JWLHS/ComfyUI-WINT8-XPU)
- [convert-to-quant (ctq)](https://github.com/newgrit1004/convert-to-quant)
```

---

```