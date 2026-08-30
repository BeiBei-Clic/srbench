# experiments/pmlb

srbench 四算法（Operon、MRGP、ITEA、FFX）在 pmlb 回归数据集上的乘性目标噪声实验。

```bash
# 结果汇总，输出 experiments/pmlb/results/pmlb_results_summary.csv
../../.venv/bin/python experiments/pmlb/pmlb_results_summary.py --input_csv experiments/pmlb/results/pmlb_results_srbench_noise*.csv --output_csv experiments/pmlb/results/pmlb_results_summary.csv
```

命令在 srbench 仓库根目录执行，完整命令见仓库根 `RUN.md`。
