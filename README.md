## `README.md`

```markdown
# ComfyUI-WINT4-XPU-beta

> **INT4 per-row 模型量化与 LoRA 推理插件 — 专为 Intel Arc A770 16GB 优化**

将扩散模型量化为 **per-row INT4 (packed uint8)**，显存节省 75%，在 A770 上裸跑 Krea2 1024×1024 出图仅需 **~8–9 GB**。
**现已支持多 LoRA 叠加**，通过自建加载链路绕过 INT4 packed shape 约束。

---

## 功能概览

| 能力 | 状态 |
|------|:--:|
| BF16/FP16/FP8/INT8 → INT4 量化 | ✅ |
| QuaRot (Hadamard 旋转) 质量提升 | ✅ **INT4 必须开启** |
| INT4 UNet 推理 (AIMDO 双路径) | ✅ |
| 单 LoRA 加载 | ✅ |
| 多 LoRA 叠加 (串联 / Stack) | ✅ |
| QKV 融合模型 LoRA (Z-image/SD3/PixArt/Flux) | ✅ |
| LoRA 参数热切换 (无残留) | ✅ |
| 7 种 LoRA key 格式自动适配 | ✅ |
| TE 量化 | ❌ 已放弃 (26 层级联误差致黑图) |

### 性能数据 (Krea2)

| 指标 | BF16 | INT8 | INT4 |
|------|:--:|:--:|:--:|
| UNet 存储 | 24 GB | 12 GB | **6 GB** |
| 推理显存 | ~24 GB | ~16–17 GB | **~8–9 GB** |
| A770 16GB 裸跑 | ❌ | ❌ | ✅ |
| LoRA | ✅ | ✅ 原生 | ✅ 自建 |

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
| `WINT4LoRALoader` | 单 LoRA，支持链式串联 (自动累加) |
| `WINT4LoRAStack` | 多 LoRA 一次叠加 (最多 5 个) |

---

## 快速开始

```
1. 量化:
   WINT4ModelQuantizer
   ├─ model_name      = 你的 BF16/FP16/INT8 模型
   ├─ model_type      = krea2 / flux2 / z-image / ...
   ├─ enable_quarot   = True  ← INT4 强烈建议开启
   ├─ device          = xpu
   └─ output_filename = my_model_int4

2. 推理 (无 LoRA):
   WINT4ModelLoader → MODEL → KSampler → VAE Decode

3. 推理 (单 LoRA):
   WINT4ModelLoader → WINT4LoRALoader (挂 1 个 LoRA) → KSampler

4. 推理 (多 LoRA 串联):
   WINT4ModelLoader → WINT4LoRALoader (LoRA 1) → WINT4LoRALoader (LoRA 2) → KSampler

5. 推理 (多 LoRA Stack):
   WINT4ModelLoader → WINT4LoRAStack (5 槽位) → KSampler
```

LoRA 强度建议 **1.5–2.0×** (INT4 反量化精度损失可能让 LoRA 效果变弱)。

---

## LoRA 支持详情

### 为什么需要自建加载器

INT4 权值的物理 shape 是 `(out_f, in_f // 2)` — 2 个 4-bit 值 packed 成 1 个 uint8。LoRA 需要 `(out_f, in_f)` 完整 shape。ComfyUI 原生 `load_lora` 走 `state_dict()` 建 key 映射，INT4 层的 packed shape 导致匹配失败。假 weight 方案又会触发 `model_patcher` 按 `numel()` 分配全量 VRAM → OOM。

### 方案

绕过 ComfyUI 原生 `load_lora`，直接读 LoRA safetensors → 解析 `lora_A`/`lora_B` (或 `lora_up`/`lora_down`) → 匹配 INT4 量化层 → 存储原始矩阵 (A=down, B=up) 在 XPU 上 → forward 时动态算 `delta = B @ A` 并原地加到 `w_dq`。

### 支持的 LoRA key 格式

| # | 格式 | 示例 |
|:-:|------|------|
| ① | Kohya 标准 | `diffusion_model.blocks.0.attn.wq.lora_B.weight` |
| ② | diffusers/simpletrainer | `transformer.blocks.0.attn.to_q.lora_B.weight` |
| ③ | SimpleTuner lycoris | `lycoris_blocks_0_attn_wq.lora_down.weight` |
| ④ | bare (无前缀) | `blocks.0.attn.wq.lora_B.weight` |
| ⑤ | onetrainer | `transformer.text_fusion.layerwise_blocks.0.attn.to_q.lora_B.weight` |
| ⑥ | legacy ComfyUI | `lora_unet_blocks_0_attn_wq.lora_down.weight` |
| ⑦ | onetrainer alt | `lora_transformer_blocks_0_attn_wq.lora_down.weight` |
| ⑧ | BFL | `single_blocks.0.attn.qkv.lora_A.weight` → 自动转换 |

后缀 `lora_B`/`lora_A` 和 `lora_up`/`lora_down` 均支持。

### 模型 LoRA 兼容矩阵

| 模型 | QKV 类型 | 状态 |
|------|---------|:--:|
| Krea2 / Qwen / HunyuanDiT 等 (~20种) | QKV 分离 | ✅ |
| Z-image / Lumina2 | QKV 融合 3 段 | ✅ |
| SD3 / PixArt | QKV 融合 3 段 | ✅ |
| Flux2 | QKV 融合 4 段 (img_attn + txt_attn) | ✅ |
| AuraFlow / HunyuanVideo | 特殊命名 | ❌ 待补 |

### QuaRot 警告

**INT4 量化必须开启 QuaRot (`enable_quarot=True`)。** INT4 只有 16 级精度，不加 Hadamard 旋转，per-channel outlier 会导致画面严重脏污。QuaRot 不是优化，是刚需。

### 多 LoRA 叠加策略

- 同 LoRA 重复加载 → `pop` 清旧数据，始终只有最新的一份
- 不同 LoRA 独立存储 → forward 分别生效
- Stack 模式 → 清空全部旧数据后重新写入

---

## 已知限制

| 限制 | 说明 |
|------|------|
| QuaRot + LoRA | 已修复 — 加载时对 A 矩阵做 group-wise 旋转 (`A_rot = A @ H^T`) |
| 仅 builtin 模式 | 不支持 ctq |
| TE 不量化 | INT4 TE 26 层级联误差致黑图, INT8 TE 出图仍黑, 已放弃 |
| 排除层保持原精度 | first/last/norm/adaLN 等不量化 |
| AuraFlow / HunyuanVideo | LoRA 命名特殊，暂不支持 |

---

## 项目演进

### 阶段 1: INT4 量化与推理 (2025-06)

- per-row INT4 量化 (amax/7, packed uint8)
- AIMDO 双路径 (开/关自动检测)
- QuaRot 支持

### 阶段 2: LoRA 加载 — 硬墙 (2025-06)

**遇到的核心矛盾:** INT4 pack shape `(out_f, in_f//2)` vs LoRA 需要 `(out_f, in_f)` — 物理不可兼得。

**尝试过的方案 (均失败):**
- 假 weight (`torch.empty(out_f, in_f)`) → `model_patcher.load()` 按 `numel()×element_size()` 分配 VBAR → 16+ GB VRAM OOM
- `as_strided` 幽灵张量 → `numel()` 仍返回完整值 → VBAR 仍 OOM
- `object.__setattr__` 隐藏 weight → `model_patcher.load()` 需要 `module.weight` 存在 → 崩溃
- 第一次 forward 时 swap → LoRA Manager 加载阶段就做 shape 检查 → 时机太晚

### 阶段 3: 自建 LoRA 加载器 (2025-06-28–29)

- 绕过 ComfyUI 原生 `load_lora`，直接读 safetensors
- 解析 `lora_up`/`lora_down` → 计算 `delta = up @ down` → 存入 `_lora_delta`
- **Bug:** 只匹配 `lora_up`/`lora_down`，Krea2 LoRA 用 `lora_A`/`lora_B` → 0 层匹配
- **修复:** 加 `lora_A`/`lora_B` 支持
- **Bug:** A/B 映射反了 → delta shape 不匹配 crash
- **修复:** `lora_A=down, lora_B=up`

### 阶段 4: 多格式适配 (2025-06-29)

- 7 种 LoRA key 格式统一 normalize
- 对照 ComfyUI 源码 `model_lora_keys_unet` 补齐所有路径
- `_normalize_layer_path()` 统一处理前缀、后缀、分隔符

### 阶段 5: 内存优化 (2025-06-29)

**问题:** 两个 LoRA 推理阶段 CPU 内存从 20GB 涨到 40–60GB, CPU 满载。

**根因分析:**
- delta 预展开 → 每层 18.9 MB fp16 → 两个 LoRA ≈ 8.5 GB
- 改为存原始 A/B 矩阵 (每层 ~600 KB → 两个 LoRA ≈ 211 MB)
- 但 A/B 存在 CPU, forward 每次 `.to(xpu)` 触发 Intel XPU 驱动分配 pinned staging buffer
- 224 层 × 20 steps × 2 次搬运 = 8960 次 staging buffer 分配, 驱动不归还 → 内存泄漏

**最终方案:**
- A/B 直接存在 XPU (fp16, ~200 MB 总量)
- Forward 里 `B @ A` 零 CPU→XPU 搬运
- 内存恢复正常

### 阶段 6: 代码清理 (2025-06-29)

- 删除 `wint4_xpu_ops.py` 中 89 行 Triton 死代码 (纯 PyTorch impl, 无 Triton kernel)
- 抽取 `wint4_lora_common.py` 消除 Loader/Stack 间 100 行重复

### 阶段 7: QKV 融合 + QuaRot 修复 + 多模型对齐 (2025-06-29)

- **QKV 融合模型支持:** Z-image/SD3/PixArt 使用融合 `.attn.qkv` 权值，LoRA 分开训练 wq/wk/wv。匹配时自动 split 为三段 slice apply
- **Flux 4 段体支持:** `double_blocks`/`single_blocks` → `blocks`，`img_attn`/`txt_attn` → `attn`
- **QuaRot LoRA 修复:** 加载时对 A 矩阵做 group-wise Hadamard 旋转 (`A_rot = A @ H^T`)，delta 自动处于旋转后空间
- **LoRA 缓存修复:** `_lora_entries` 从 list 改为按 `lora_name` 索引的 dict，加载前 `pop` 清旧数据，杜绝残留
- **多模型 key 对齐:** 补全 `attention.out`→`attn.wo`、`q_proj`→`wq`、`self_attn.q`→`attn.wq` 等映射

---

## Bug 修复记录

| # | Bug | 修复 |
|---|------|------|
| 1 | 花屏 | key 命名对齐 |
| 2 | 偏色 | 排除列表同步 ctq |
| 3 | 元数据不识别 | 双读 quarot/convrot |
| 4 | 缺 input_scale | 量化器写入 |
| 5 | unexpected key | `object.__setattr__` 绕过 nn.Module |
| 6 | bias device mismatch | `.to(device=x.device)` |
| 7 | 循环导入 | import 移方法内 |
| 8 | ops 被覆盖 | 恢复 Int4XPUOps |
| 9 | AIMDO 显存泄漏 | cast/uncast 闭环 + empty_cache |
| 10 | `quint4x2` 创建失败 | uint8 手动 pack |
| 11 | `bitwise_and_xpu` BFloat16 | `weight.to(torch.uint8)` |
| 12 | 假 weight VBAR OOM | 放弃, 改 `_lora_entries` |
| 13 | `load_text_encoder` 不存在 | → `load_clip(ckpt_paths=[...])` |
| 14 | TE embed/lm_head shape mismatch | 通用 `_SKIP_KEYWORDS` |
| 15 | swap 方案 Linear 无 weight | 放弃 swap |
| 16 | `MODEL_TYPES` ImportError | 恢复正确文件 |
| 17 | `clip_type` 传字符串无效 | 映射到 `CLIPType` 枚举 |
| 18 | INT4 TE 26 层级联 → 黑图 | TE 改用 INT8 (后也放弃) |
| 19 | LoRA 匹配 0 层 | 加 `lora_A`/`lora_B` 支持 |
| 20 | delta 矩阵乘法 shape 错误 | A/B 的 up/down 角色对调 |
| 21 | 不同 LoRA shape 累加崩溃 | 列表独立存储 |
| 22 | CPU 内存 40–60 GB + 满载 | A/B 存 XPU, 零搬运 |
| 23 | Z-image LoRA 不生效 (QKV 融合) | QKV fallback + slice apply |
| 24 | QuaRot ON 时 LoRA 效果弱/无 | A 矩阵加载时 group-wise 旋转 |
| 25 | LoRA 调整参数后效果残留 | `_lora_entries` 改 dict + pop |
| 26 | Z-image `attention.out` 无法匹配 | `.attn.out` → `.attn.wo` |

---

## 文件清单

```
ComfyUI-WINT4-XPU-beta/
├── __init__.py                  # 节点注册 (4 节点)
├── wint4_model_quantizer.py     # 量化 (BF16/FP16/FP8/INT8 → INT4 + QuaRot)
├── wint4_model_loader.py        # 加载 INT4 UNet
├── wint4_xpu_ops.py             # 推理 ops (AIMDO 双路径 + LoRA delta)
├── wint4_lora_loader.py         # 单 LoRA 加载 (自建链路)
├── wint4_lora_stack.py          # 多 LoRA 一次叠加 (最多 5 个)
├── wint4_lora_common.py         # 共享: key 格式归一化函数
├── wint8_quarot.py              # Hadamard 旋转
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 未来展望

| 优先级 | 方向 | 说明 |
|:---:|------|------|
| 🟡 | LoRA Unloader 节点 | 手动释放 `_lora_entries`, 供 session 内切换 LoRA |
| 🟡 | AuraFlow / HunyuanVideo LoRA | 补全特殊命名模型的 key 映射 |
| 🟢 | INT4 TE 重试 | 视觉层排除 + 低精度层混合, 缓解 26 层级联误差 |
| 🟢 | FB Cache | hook Krea2 first block 前向, 缓存跳过 |
| 🟢 | PR 给 ComfyUI | `model_patcher.py` 一行让 uint8 参数按实际字节分配 VRAM |
| 🔵 | OpenVINO 调研 | 看 `comfyui-openvino` 源码能否对接 |

---

## 相关链接

- **本插件**: https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta
- **INT8 插件**: https://github.com/JWLHS/ComfyUI-WINT8-XPU

---

## 鸣谢

本插件全部由 **DeepSeek V4 Pro** 完成。

开发过程中参考了以下开源项目：

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — LoRA key 映射机制（`comfy/lora.py`、`comfy/lora_convert.py`）与模型对齐方案（`comfy/utils.py` 中 `_to_diffusers` 系列函数）
- **[ComfyUI-WINT8-XPU](https://github.com/JWLHS/ComfyUI-WINT8-XPU)** — 兄弟项目，排除列表（`_EXCLUSIONS`）最初来源
- **[convert-to-quant (ctq)](https://github.com/newgrit1004/convert-to-quant)** — INT8/INT4 量化工具链，Hadamard 旋转代码（`wint8_quarot.py`）及排除列表同步来源
```