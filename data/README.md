# data

训练数据相关 json/jsonl、reform 文件、split 文件和标准测试登记索引暂存于此。

当前登记：

- `dermoinstruct_mcqa_train_10k.json`: 从 DermoInstruct 训练侧四选一 MCQA 子集派生的 10,000 条训练样本。
- `dermoinstruct_closed_vqa_test_2k.json`: 从同一来源派生、与 10k 训练子集按 `id` 去重的 2,000 条闭集 VQA 测试样本。
- `MANIFEST.md`: `data/` 目录下所有派生数据文件的统一 manifest。

不要在此目录存放原始图像、预训练权重、checkpoint 或大体量二进制产物。
暂时不要移动或改写 `dataset_final/benchmark/`。
