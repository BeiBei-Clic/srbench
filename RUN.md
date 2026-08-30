# PMLB 噪声实验

4 个 SR 算法（Operon、MRGP、ITEA、FFX，均为 srbench tuned 配置）在本地 pmlb/datasets
全部回归数据集上的乘性目标噪声实验。噪声施加在训练集目标上：y*(1+xi)，xi~N(0,strength^2)。

```bash
# 冒烟：2 个数据集 x 快算法 x 单噪声档，验证流程与 CSV 落盘
.venv/bin/python experiments/pmlb/pmlb_noise_batch.py --dataset_limit 2 --algorithms ffx,tuned.OperonRegressor --noises 0.1 --max_rows 100 --jobs 4

# 正式全量：全部回归数据集 x 4 算法 x 噪声档 0/0.001/0.01/0.1（默认前 200 行、56 并行）
.venv/bin/python experiments/pmlb/pmlb_noise_batch.py

# 结果整合：全量明细 + 按算法×噪声拆分 16 份明细 + summary 一键生成
.venv/bin/python experiments/pmlb/pmlb_consolidate.py
```

- 换噪声档：改 `--noises`（逗号分隔列表）；换噪声种子：`--noise_seed`；换划分种子：`--seed`
- 断点续跑：直接重跑同命令，已有 (dataset, algorithm) 结果自动跳过，只追加不覆盖
- 结果按噪声档落 `experiments/pmlb/results/pmlb_results_srbench_noise{强度}.csv`，表达式列固定在最后
