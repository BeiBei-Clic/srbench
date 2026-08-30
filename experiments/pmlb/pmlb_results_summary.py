"""
汇总批量噪声实验的多份结果 CSV（{基础名}_noise{强度}.csv）为分组统计表。

按数据集名前缀分 Feynman / Strogatz / Black-box 三组，按 算法 x 组 统计：
  - r2 负数、nan、inf 一律按 0 参与均值/标准差
  - r2_valid_count 只计原始 r2 有限且 >= 0 的样本
  - recovery_rate = 原始 r2 > 0.9 的样本数 / total_count
  - complexity、seconds 只在 status=ok 且为有限数值时统计
  - 标准差用总体口径 ddof=0

用法（在 srbench 仓库根目录）:
  .venv/bin/python experiments/pmlb/pmlb_results_summary.py \
      --input_csv experiments/pmlb/results/pmlb_results_srbench_noise0.0.csv \
                  experiments/pmlb/results/pmlb_results_srbench_noise0.001.csv \
                  experiments/pmlb/results/pmlb_results_srbench_noise0.01.csv \
                  experiments/pmlb/results/pmlb_results_srbench_noise0.1.csv \
      --output_csv experiments/pmlb/results/pmlb_results_summary.csv
"""
import re
import argparse
import numpy as np
import pandas as pd

from complexity_unified import unified_complexity

parser = argparse.ArgumentParser(description='pmlb 噪声实验结果汇总')
parser.add_argument('--input_csv', nargs='+', required=True)
parser.add_argument('--output_csv', type=str,
                    default='experiments/pmlb/results/pmlb_results_summary.csv')
args = parser.parse_args()


def group_of(dataset):
    if dataset.startswith('feynman_'):
        return 'Feynman'
    if dataset.startswith('strogatz_'):
        return 'Strogatz'
    return 'Black-box'


rows = []
for path in args.input_csv:
    m = re.search(r'_noise([\d.]+)\.csv$', path)
    noise_strength = float(m.group(1))
    df = pd.read_csv(path)
    df = df[df['status'] != 'skip']  # 未实际运行的样本不纳入统计
    for (dataset_group, algorithm), g in df.groupby(
            [df['dataset'].map(group_of), 'algorithm']):
        r2_raw = pd.to_numeric(g['r2'], errors='coerce')
        r2_clean = r2_raw.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)

        ok = g[g['status'] == 'ok']
        # 统一复杂度口径：表达式树节点数（运算符/变量/常数各 1），从 expr 解析
        comp = pd.Series([unified_complexity(e) for e in ok['expr']],
                         dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        secs = pd.to_numeric(ok['seconds'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()

        rows.append({
            'noise_strength': noise_strength,
            'group': dataset_group,
            'algorithm': algorithm,
            'r2_mean': round(r2_clean.mean(), 4),
            'r2_std': round(r2_clean.std(ddof=0), 4),
            'r2_valid_count': int((r2_raw >= 0).sum()),
            'total_count': len(g),
            'recovery_rate': round(float((r2_raw > 0.9).sum()) / len(g), 4),
            'complexity_mean': round(comp.mean(), 2) if len(comp) else '',
            'complexity_std': round(comp.std(ddof=0), 2) if len(comp) else '',
            'complexity_count': len(comp),
            'seconds_mean': round(secs.mean(), 1) if len(secs) else '',
            'seconds_std': round(secs.std(ddof=0), 1) if len(secs) else '',
            'seconds_count': len(secs),
        })

GROUP_ORDER = ['Feynman', 'Strogatz', 'Black-box']

out = pd.DataFrame(rows)
out['group'] = pd.Categorical(out['group'], categories=GROUP_ORDER, ordered=True)
out = out.sort_values(['noise_strength', 'group', 'algorithm']).reset_index(drop=True)
out['group'] = out['group'].astype(str)
out.to_csv(args.output_csv, index=False)
print(out.to_string(index=False))
print(f'\n汇总已保存: {args.output_csv}')
