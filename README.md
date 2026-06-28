```markdown
# WINT4 XPU INT4 量化插件 — 实验版

> **⚠️ 实验版 (Beta)**：功能正常但尚未经过全面测试。当前版本 **不支持 LoRA**（详见[已知限制](#已知限制)）。
>
> 如需 LoRA + 量化 → 使用 [ComfyUI-WINT8-XPU](https://github.com/JWLHS/ComfyUI-WINT8-XPU)

---

## 目录

1. [功能概览](#功能概览)
2. [安装](#安装)
3. [节点详解](#节点详解)
   - [WINT4 Model Quantizer](#wint4-model-quantizer)
   - [WINT4 Model Loader](#wint4-model-loader)
4. [INT4 vs INT8 对比](#int4-vs-int8-对比)
5. [AIMDO DynamicVRAM 使用建议](#aimdo-dynamicvram-使用建议)
6. [完整工作流](#完整工作流)
7. [支持模型](#支持模型)
8. [常见问题](#常见问题)
9. [已知限制](#已知限制)
10. [文件清单](#文件清单)
11. [相关链接](#相关链接)

---

## 功能概览

将 **BF16 / FP16 / FP8 / INT8** 扩散模型量化为 **per-row INT4（packed uint8）**：

| 指标 | 效果 |
|------|------|
| **显存（存储）** | ~25% of BF16（≈6 GB vs 24 GB） |
| **推理显存 (Krea2 1024×1024)** | ~8–9 GB |
| **推理速度** | 接近 INT8，明显快于 BF16 原版 |
| **画质** | 正常，无花屏 |
| **加载** | < 2 秒 |
| **AIMDO** | ✅ 兼容（自动检测，双路径） |
| **LoRA** | ❌ 当前不支持 |

### 原理

```
量化阶段：                 推理阶段：
                            
BF16/INT8 权重              packed uint8 (out_f, in_f//2)
    │                           │
    ▼                           ▼
amax / 7 = scale            AIMDO: cast_bias_weight → XPU
round → [-8,7]              无 AIMDO: 局部变量 → XPU
pack → uint8                     │
    │                           ▼
    ▼                       unpack → 解出 2 路 4-bit
safetensors                  → fp16 dequant
                            → F.linear
                            → empty_cache() 回收
```

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta.git
```

依赖同 INT8 插件（`convert-to-quant` + `safetensors`），ComfyUI 启动时自动安装。

---

## 节点详解

---

### WINT4 Model Quantizer

**位置：** `WINT4` → `WINT4 Model Quantizer`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `model_name` | 下拉 | — | BF16/FP16/FP8/INT8 模型 |
| `model_type` | 下拉 | `flux2` | 架构类型，控制排除列表 |
| `enable_quarot` | 开关 | `False` | Hadamard 旋转，提升质量 |
| `group_size` | 整数 | `128` | QuaRot 分组大小 |
| `device` | 下拉 | `xpu` | 量化计算设备 |
| `output_filename` | 文本 | `model_int4` | 输出至 `ComfyUI/output/` |

#### 支持的输入格式

| 输入 dtype | 处理方式 |
|:---|------|
| BF16 / FP16 / FP32 | 直接 → float32 → INT4 |
| FP8 | `.float()` → float32 → INT4 |
| **INT8** | 读取 `weight_scale` 反量化 → float32 → INT4 |

#### QuaRot (enable_quarot)

勾选后对每组权重应用 Hadamard 正交旋转，将浮点 outlier 均匀分散 → 量化误差更低。`group_size` 需整除 `in_features`。

---

### WINT4 Model Loader

**位置：** `WINT4` → `WINT4 Model Loader`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|:---:|------|
| `unet_name` | 下拉 | — | INT4 量化后的 `.safetensors` |
| `model_type` | 下拉 | `flux2` | **必须和量化时一致** |

输出 `MODEL`，可直接接入 KSampler。

---

## INT4 vs INT8 对比

| | INT8 | INT4 |
|---|---|---|
| **存储 vs BF16** | 50% | **25%（再省一半）** |
| **推理显存 (Krea2)** | ~16-17 GB | **~8-9 GB** |
| **A770 16GB 能否裸跑** | ❌ 常超 | ✅ 够 |
| **推理速度** | ≈ BF16 | 接近 INT8 |
| **模型输入** | BF16/FP16/FP8 | BF16/FP16/FP8/**INT8** |
| **AIMDO** | ✅ | ✅ |
| **LoRA** | ✅ | ❌ |
| **QuaRot** | ✅ (builtin) | ✅ |
| **ctq** | ✅ (auto/ctq) | ❌ |

---

## AIMDO DynamicVRAM 使用建议

| 场景 | 建议 |
|------|------|
| INT4 裸跑 (A770 16GB) | **关 AIMDO** — 8-9 GB 够用 |
| INT4 + 大分辨率 | 开 AIMDO — 自动检测，走闭环 |
| BF16 原版 | 用 INT8 插件 + AIMDO |

AIMDO 开启时：`cast_bias_weight` → unpack → dequant → `uncast_bias_weight`，共享显存正常释放。

---

## 完整工作流

```
1. 量化:
   WINT4ModelQuantizer:
     model_name      = BF16 / INT8 模型
     model_type      = krea2 / flux2 / ...
     device          = xpu
     output_filename = my_model_int4

2. 推理:
   WINT4ModelLoader (unet_name + model_type) → MODEL
   XPU AIMDO Status: Enable_DynamicVRAM = OFF (推荐)
   KSampler → VAE Decode → 出图
```

---

## 支持模型

与 INT8 插件共享相同的排除列表（从 convert-to-quant 同步）：

```
flux2   z-image   chroma   wan   ltx2
qwen    ernie    hidream   boogu
krea2   ideogram4   auto
```

排除层（保持 BF16/FP16）：`img_in`、`txt_in`、`final_layer`、`adaLN`、`norm_*`、`patch_embedding` 等。

---

## 常见问题

### Q: 加载时看到 `[WARNING] unet unexpected: ['int4_model_type', 'int4_quantized', ...]`？

**A:** 正常。元数据 key 不被模型消费，无害。

### Q: INT8 → INT4 量化后文件体积差多少？

**A:** INT8 12GB → INT4 ~6GB，体积减半。量化器自动读 INT8 的 `weight_scale` 反量化，无需手动操作。

### Q: 能和 LoRA 一起用吗？

**A:** 不能。INT4 packed shape（`out_f, in_f//2`）与 LoRA 期望的完整 shape 不兼容。INT4 层会打印 ERROR 但推理正常。排除层（first/last/norm）仍可用 LoRA。

---

## 已知限制

| 限制 | 说明 |
|------|------|
| **LoRA 不支持** | 当前 INT4 层不做 LoRA，排除层（first/last/norm）除外 |
| **仅 builtin 模式** | 不支持 ctq（ctq 没有 INT4 路径） |
| **排除层保持原精度** | first/last/norm 等不量化，推理时用更大显存 |
| **large resolution 需 AIMDO** | 超大分辨率建议开 AIMDO DynamicVRAM |

---

## 文件清单

```
ComfyUI-WINT4-XPU/
├── __init__.py                  # 节点注册
├── wint4_model_quantizer.py     # 量化（BF16/FP16/FP8/INT8 → INT4 + QuaRot）
├── wint4_model_loader.py        # 加载
├── wint4_xpu_ops.py             # 推理 ops（AIMDO 双路径 + fp16 unpack）
├── wint8_quarot.py              # Hadamard 旋转（与 INT8 共享）
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 相关链接

- **本插件 (INT4)**：https://github.com/JWLHS/ComfyUI-WINT4-XPU-beta
- **INT8 插件**：https://github.com/JWLHS/ComfyUI-WINT8-XPU
```

---

