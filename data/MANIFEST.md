# data manifest

本文件是 `data/` 目录下所有派生数据文件的统一登记表。

## 当前数据清单

| 数据文件 | 状态 | 用途 | 样本数 | 来源 |
|---|---|---|---:|---|
| `dermoinstruct_mcqa_train_10k.json` | PASS | 训练侧四选一 diagnosis MCQA 子集 | 10,000 | DermoInstruct-MCQA-subset |
| `dermoinstruct_closed_vqa_test_2k.json` | PASS | 闭集四选一 VQA 测试子集 | 2,000 | DermoInstruct-MCQA-subset |

## dermoinstruct_mcqa_train_10k.json

### 基本信息

| 字段 | 内容 |
|---|---|
| 派生文件 | `/mnt/nas1/disk06/bowenguo/codes/DermoGPT/data/dermoinstruct_mcqa_train_10k.json` |
| 源 annotation | `/mnt/nas1/disk06/bowenguo/datasets/VQA/DermoInstruct/derived_json_versions/DermoInstruct-MCQA-subset.json` |
| 图像根目录 | `/mnt/nas1/disk06/bowenguo/datasets/VQA/DermoInstruct` |
| 数据类型 | DermoInstruct training-side diagnosis 4-way MCQA |
| 数据格式 | LLaVA/Qwen-VL JSON array: `id`, `image`, `conversations[human,gpt]` |
| 备注 | 该文件是训练侧 DermoInstruct 子集，不是 DermoBench benchmark 数据。 |

### 抽样方法

| 字段 | 内容 |
|---|---:|
| 方法 | deterministic stratified sampling by ground-truth diagnosis |
| 随机种子 | 20260722 |
| 目标样本数 | 10,000 |
| 源文件样本数 | 165,700 |
| 源文件正确诊断标签数 | 214 |
| 子集正确诊断标签数 | 214 |
| 每个正确诊断标签最低保留 | 1 |

### 校验结果

| 校验项 | 结果 |
|---|---:|
| 样本数 | 10,000 |
| 每条选项数 | 4 |
| bad option rows | 0 |
| missing images | 0 |
| 是否修改 `dataset_final/benchmark/` | 否 |

### 正确选项位置分布

| 选项 | 数量 |
|---|---:|
| A | 2,550 |
| B | 2,496 |
| C | 2,499 |
| D | 2,455 |

### 图像来源分布

| 来源 | 数量 |
|---|---:|
| isic | 6,327 |
| dermnet | 1,077 |
| f17k | 695 |
| daffodil | 548 |
| pumch | 532 |
| sd198 | 392 |
| passion | 287 |
| midas | 142 |

### Top Ground-Truth Diagnosis

| 诊断 | 数量 |
|---|---:|
| Nevus / Mole / Melanocytic Nevus | 2,273 |
| Benign Lesion | 1,663 |
| Basal Cell Carcinoma | 778 |
| Melanoma | 594 |
| Eczema / Dermatitis | 403 |
| Atypical Melanocytic Lesion / Spitzoid Lesion | 355 |
| Seborrheic Keratosis | 346 |
| Psoriasis | 303 |
| Squamous Cell Carcinoma | 191 |
| Fungal Infection | 185 |
| Actinic Keratosis | 182 |
| Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS/TEN) | 177 |
| Acne | 177 |
| Vitiligo | 149 |
| Scabies / Lyme Disease / Infestations and Bites | 149 |
| Lichen Planus | 93 |
| Rosacea | 80 |
| Other / Not Specified | 76 |
| Psoriasis / Lichen Planus | 74 |
| Malignant Lesion / Skin Cancer / Malignant Neoplasm | 73 |
| Nail Disorder | 72 |
| Morphea / Scleroderma | 71 |
| Lentigo / Solar Lentigo / Actinic Pigmentation | 69 |
| Viral Infections | 68 |
| Acne / Rosacea | 61 |
| Dermatofibroma / Fibroma / Fibrous Papule | 58 |
| Vascular Tumors | 54 |
| Squamous Cell Carcinoma In Situ | 49 |
| Systemic Disease | 41 |
| Light Diseases / Disorders of Pigmentation | 39 |

## dermoinstruct_closed_vqa_test_2k.json

### 基本信息

| 字段 | 内容 |
|---|---|
| 派生文件 | `/mnt/nas1/disk06/bowenguo/codes/DermoGPT/data/dermoinstruct_closed_vqa_test_2k.json` |
| 源 annotation | `/mnt/nas1/disk06/bowenguo/datasets/VQA/DermoInstruct/derived_json_versions/DermoInstruct-MCQA-subset.json` |
| 图像根目录 | `/mnt/nas1/disk06/bowenguo/datasets/VQA/DermoInstruct` |
| 数据类型 | closed-set diagnosis 4-way VQA test subset |
| 数据格式 | LLaVA/Qwen-VL JSON array: `id`, `image`, `conversations[human,gpt]` |
| 备注 | 该文件用于本地闭集 VQA 测试；来源为 DermoInstruct 训练侧 MCQA，不等同于 DermoBench benchmark。 |

### 抽样方法

| 字段 | 内容 |
|---|---:|
| 方法 | exclude `dermoinstruct_mcqa_train_10k.json` ids, then deterministic stratified sampling by ground-truth diagnosis |
| 随机种子 | 20260723 |
| 目标样本数 | 2,000 |
| 源文件样本数 | 165,700 |
| 排除训练子集 id 数 | 10,000 |
| 剩余候选样本数 | 155,700 |
| 源文件正确诊断标签数 | 214 |
| 剩余候选正确诊断标签数 | 213 |
| 子集正确诊断标签数 | 213 |
| 与 `dermoinstruct_mcqa_train_10k.json` 的 id 重叠 | 0 |

说明：`Foreign Body Reaction` 在源文件中仅有 1 条，已被 10k 训练子集抽中；排除训练 id 后该标签无剩余候选，因此 2k 测试子集覆盖 213/214 个源标签。

### 校验结果

| 校验项 | 结果 |
|---|---:|
| 样本数 | 2,000 |
| 每条选项数 | 4 |
| bad option rows | 0 |
| missing images | 0 |
| 是否修改 `dataset_final/benchmark/` | 否 |

### 正确选项位置分布

| 选项 | 数量 |
|---|---:|
| A | 529 |
| B | 502 |
| C | 481 |
| D | 488 |

### 图像来源分布

| 来源 | 数量 |
|---|---:|
| isic | 1,264 |
| dermnet | 203 |
| sd198 | 128 |
| f17k | 119 |
| daffodil | 104 |
| pumch | 99 |
| passion | 50 |
| midas | 33 |

### Top Ground-Truth Diagnosis

| 诊断 | 数量 |
|---|---:|
| Nevus / Mole / Melanocytic Nevus | 454 |
| Benign Lesion | 332 |
| Basal Cell Carcinoma | 155 |
| Melanoma | 118 |
| Eczema / Dermatitis | 80 |
| Atypical Melanocytic Lesion / Spitzoid Lesion | 70 |
| Seborrheic Keratosis | 68 |
| Psoriasis | 60 |
| Squamous Cell Carcinoma | 37 |
| Fungal Infection | 36 |
| Actinic Keratosis | 35 |
| Stevens-Johnson Syndrome / Toxic Epidermal Necrolysis (SJS/TEN) | 34 |
| Acne | 34 |
| Vitiligo | 29 |
| Scabies / Lyme Disease / Infestations and Bites | 29 |
| Lichen Planus | 18 |
| Rosacea | 15 |
| Psoriasis / Lichen Planus | 14 |
| Malignant Lesion / Skin Cancer / Malignant Neoplasm | 14 |
| Other / Not Specified | 14 |
| Morphea / Scleroderma | 13 |
| Viral Infections | 13 |
| Lentigo / Solar Lentigo / Actinic Pigmentation | 13 |
| Nail Disorder | 13 |
| Acne / Rosacea | 11 |
| Dermatofibroma / Fibroma / Fibrous Papule | 11 |
| Vascular Tumors | 10 |
| Squamous Cell Carcinoma In Situ | 9 |
| Light Diseases / Disorders of Pigmentation | 7 |
| Hyperpigmentation | 7 |

## qwen35_mcqa_format_sft_train_10k.json
- Created: 2026-07-23T12:10:42
- Source: `data/dermoinstruct_mcqa_train_10k.json`
- Rows: 10000
- Purpose: Qwen3.5-4B MCQA format-calibration SFT data before GRPO.
- User content: original DermoInstruct MCQA question/options plus the standard prompt suffix, byte-preserved in `scripts/build_qwen35_format_sft_data.py`.
- Assistant content: strict `<think>...</think>
<answer>X</answer>` with a short format-calibration rationale and a single-letter answer.
- Constraints: no system role, preserves `image` and original sample metadata outside `conversations`, derived under `data/` only.

## qwen35_mcqa_option_sft_train_10k.json
- Created: 2026-07-23T21:22:20
- Source: `data/dermoinstruct_mcqa_train_10k.json`
- Rows: 10000
- Purpose: Qwen3.5-4B dense LoRA SFT MCQA clean-base control; option-only target for direct answer training.
- User content: original DermoInstruct MCQA question/options plus only `Answer with only the single correct option letter.`.
- Assistant content: one uppercase option letter only, matching `^[A-D]$`; no tags, explanation, whitespace, newline, or punctuation.
- Letter counts: A=2550, B=2496, C=2499, D=2455.
- Constraints: preserves `image` and original sample metadata outside `conversations`, derived under `data/` only, does not touch `dataset_final/benchmark/`.

