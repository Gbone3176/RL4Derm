# DermoGPT Test Results Summary

本文件按数据集分类记录已产生完整测试产物的评测/推理结果。`valid` 表示测试已完成且可按记录的设置比较；`invalid` 表示有完整产物但存在模板、解析或其他可比性问题；`incomplete` 表示未完成或尚未触发。只有设置兼容的 `valid` 条目可直接横向比较。

## 记录规范

每条记录对应一个实际测试目录，并应保留 `test.log`、`predictions.jsonl`、`metrics.json`/`summary.json`、参数快照和预检证据。指标中的 `invalid` 是无法解析为允许答案的样本数；`strict_single_letter_rate` 以各测试目录的原始统计口径为准。

`Training method` 表示被测模型的任务适配来源：`Zeroshot` 为未进行任务微调的基础模型，`SFT` 为监督微调模型，`GRPO` 为经 GRPO 强化学习训练的模型。

## DermInstruct / Closed VQA

| Run ID | Status | Training method | Model / checkpoint | Samples | Accuracy | Invalid | Parse status | Evidence | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| Ts36 | valid | SFT | Qwen3.5-4B + Tr43 checkpoint-2500 | 2000 | 0.0000 (0/2000) | 2000 | `no_parse:2000` | `experiments/Qwen3.5-4B/Ts/Ts36_qwen35_sft_lora_mcqa_closed_vqa_2k_acc_eval_sft_template` | Template-consistent execution; model outputs were not parseable |
| Ts38 | invalid | SFT | Qwen3.5-4B + Tr43 checkpoint-2500 | 20 | 0.0000 (0/20) | 18 | `no_parse:18`, `multiple_candidate_letters:1`, `single_standalone_letter:1` | `experiments/Qwen3.5-4B/Ts/Ts38_qwen35_sft_lora_mcqa_closed_vqa_20_sample_infer_sft_template` | Regression smoke; not a formal full-test result |
| Ts39 | valid | SFT | Qwen3.5-4B + Tr43 checkpoint-2500 | 20 | 0.5000 (10/20) | 0 | `strict_single_letter:20` | `experiments/Qwen3.5-4B/Ts/Ts39_qwen35_sft_lora_load_bugfix_20sample_regression` | Loader regression smoke; strict single-letter rate 1.0 |
| Ts48 | valid | SFT | Qwen3.5-4B + Tr43 checkpoint-2500 | 2000 | 0.8760 (1752/2000) | 0 | `strict_single_letter:2000` | `experiments/Qwen3.5-4B/Ts/Ts48_shared` | Greedy, `max_new_tokens=8`, strict single-letter rate 1.0 |
| Ts49 | valid | Zeroshot | Qwen3.5-4B base | 2000 | 0.5580 (1116/2000) | 0 | `option_prefix:1762`, `single_standalone_letter:1`, `strict_single_letter:237` | `experiments/Qwen3.5-4B/Ts/Ts49_qwen35_4b_base_zero_shot_closed_vqa_2k` | Base zero-shot baseline; strict single-letter rate 0.1185 |

## DermMaskTriads / Image-Disjoint

| Run ID | Status | Training method | Model / checkpoint | Samples | Accuracy | Invalid | Parse status | Evidence | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| Ts65 | valid | Zeroshot | HuatuoGPT-Vision-7B base | 2000 | 0.4870 (974/2000) | 37 | `strict_single_letter:1481`, `option_prefix:482`, `no_parse:33`, `prefix_not_allowed:4` | `experiments/HuatuoGPT-Vision-7B/Ts/Ts65_huatuogpt_vision_7b_dermmask_image_disjoint_zeroshot_after_Tr52_retry_local_clip_20260729` | Renamed from duplicate Ts54; full image-disjoint test; retain raw outputs; parser supports labels beyond A-D |
| Ts58 | invalid | SFT | HuatuoGPT-Vision-7B + SFT checkpoint-3422 | 150 | 0.0000 (0/150) | 150 | `no_parse:150` | `experiments/HuatuoGPT-Vision-7B/Ts/Ts58_huatuogpt_vision_7b_sft_dermmask_image_disjoint_ckpt3422` | SFT evaluation produced no parseable predictions |
| Ts60 | valid | SFT | HuatuoGPT-Vision-7B + SFT checkpoint-3422 | 20 | 0.6000 (12/20) | 1 | `single_standalone_allowed_letter:19`, `multiple_candidate_letters:1` | `experiments/HuatuoGPT-Vision-7B/Ts/Ts60_huatuogpt_vision_7b_sft_dermmask_decodefix_smoke20_ckpt3422_retry_gpu4` | Decode-fix smoke only; not a full-test comparison |
| Ts54 | incomplete | Zeroshot | HuatuoGPT-Vision-7B base | 2000 | pending | pending | pending | `experiments/HuatuoGPT-Vision-7B/Ts/Ts54_huatuogpt_vision_7b_dermmask_image_disjoint_zeroshot_after_Tr52` | Original watcher directory; static gate passed, runtime evaluation had not started |

## DermMaskTriads / Four-Choice Image-Disjoint

固定测试清单为 `data/dermmask_qwen35_4choice_test_image_disjoint.json`（1000 条、每条 4 个选项，SHA-256=`6e14ef2dbff6028757488c8ccb55dbf43cf4b49e595341badc91d4e048001440`）。只有 `valid` 条目可直接横向比较。

| Run ID | Status | Training method | Model / checkpoint | Samples | Accuracy | Invalid | Parse status | Evidence | Notes |
|---|---|---|---|---:|---:|---:|---|---|---|
| Ts64 | invalid | Zeroshot | DermoGPT-RL base | 1000 | 0.9820† (982/1000) | 0 | `option_prefix:1000`† | `experiments/DermoGPT-RL/Ts/Ts64_dermogpt_rl_dermmask_4choice_image_disjoint_test` | Native inference produced 1000 raw outputs, but the post-processing step failed with `SyntaxError`; metric is a manual audit of `native_outputs/*.jsonl`, not a formal comparison result |
| Ts55 | valid | Zeroshot | HuatuoGPT-Vision-7B base | 1000 | 0.8300 (830/1000) | 0 | `strict_single_letter:522`, `option_prefix:478` | `experiments/HuatuoGPT-Vision-7B/Ts/Ts55_huatuogpt_vision_7b_dermmask_4choice_image_disjoint_zeroshot_after_Tr52_retry_local_clip_20260730` | Four-choice image-disjoint zero-shot baseline |
| Ts61 | valid | SFT | HuatuoGPT-Vision-7B + Tr59 LoRA checkpoint | 1000 | 0.9740 (974/1000) | 2 | `single_standalone_allowed_letter:998`, `multiple_candidate_letters:2` | `experiments/HuatuoGPT-Vision-7B/Ts/Ts61_huatuogpt_vision_7b_sft_lora_tr59_4choice_image_disjoint_test_1k` | Four-choice image-disjoint SFT result; invalid count retained from metrics |
| Ts63 | valid | SFT | Qwen3.5-9B + Tr62 LoRA checkpoint | 1000 | 0.9760 (976/1000) | 3 | `strict_single_letter:997`, `no_parse:3` | `experiments/Qwen3.5-9B/Ts/Ts63_qwen35_9b_sft_lora_tr62_4choice_image_disjoint_test_1k` | Four-choice image-disjoint SFT result; strict single-letter rate 0.997 |

† Ts64 的准确率和解析统计仅由原始 native 输出逐条复核得到；由于官方后处理失败，该条保持 `invalid`，不得纳入正式横向比较。

## Comparison Notes

- Ts54/Ts65 are full mixed-choice image-disjoint evaluations, while Ts55/Ts61/Ts63 are the fixed four-choice image-disjoint group; compare only within matching dataset construction and parser settings.
- Ts64 is retained as an audit record but is excluded from formal comparison until its post-processing pipeline is repaired and rerun.
- Ts36/Ts38/Ts58 are retained for auditability, but their parse failures make them unsuitable as model-quality comparisons.
- Raw outputs and per-sample predictions remain in each registered run directory; this file is an index, not a replacement for those artifacts.
- Historical run paths were migrated to the canonical model/type/run layout on 2026-08-03; the active Tr58 process retains a temporary legacy alias until completion.
