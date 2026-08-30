"""
整合四算法噪声实验结果，产出（均在 experiments/pmlb/results/）：

1. pmlb_results_srbench_all.csv       全量明细，四份噪声 CSV 合并，首列 noise_strength
2. {算法}_results/                    每算法一个文件夹：
     {算法}_noise{σ}.csv  四份明细（每个数据集一行，表达式列最后）
     {算法}_summary.csv    该算法的分组汇总（noise_strength × 数据集族，pmlb4 口径）
3. pmlb_results_summary.csv           全局汇总（复用 pmlb_results_summary 口径）

用法（在 srbench 仓库根目录）:
  .venv/bin/python experiments/pmlb/pmlb_consolidate.py
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import subprocess
import sys

from complexity_unified import unified_complexity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

parser = argparse.ArgumentParser(description='整合噪声实验结果')
parser.add_argument('--input_glob', type=str,
                    default=os.path.join(RESULTS_DIR, 'pmlb_results_srbench_noise*.csv'))
parser.add_argument('--results_dir', type=str, default=RESULTS_DIR)
args = parser.parse_args()

ALGO_SHORT = {'OperonRegressor': 'Operon', 'MRGPRegressor': 'MRGP',
              'ITEARegressor': 'ITEA', 'ffx': 'FFX'}

# 读入四份噪声档明细，从文件名解析噪声强度
frames = []
for path in sorted(glob.glob(args.input_glob)):
    m = re.search(r'_noise([\d.]+)\.csv$', path)
    noise = float(m.group(1))
    df = pd.read_csv(path)
    df.insert(0, 'noise_strength', noise)
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

all_df['complexity_unified'] = [unified_complexity(e) for e in all_df['expr']]

DETAIL_COLS = ['noise_strength', 'dataset', 'algorithm', 'status', 'n_features',
               'r2', 'rmse', 'complexity', 'complexity_unified', 'seconds', 'error', 'expr']

# 1. 全量明细
all_path = os.path.join(args.results_dir, 'pmlb_results_srbench_all.csv')
all_df[DETAIL_COLS].sort_values(['dataset', 'algorithm', 'noise_strength']).to_csv(all_path, index=False)


def group_of(dataset):
    if dataset.startswith('feynman_'):
        return 'Feynman'
    if dataset.startswith('strogatz_'):
        return 'Strogatz'
    return 'Black-box'


# 2. 每算法一个文件夹：四份明细 + 一份 summary
for algo, short in ALGO_SHORT.items():
    folder = os.path.join(args.results_dir, f'{short}_results')
    os.makedirs(folder, exist_ok=True)
    sub = all_df[all_df.algorithm == algo]

    for noise, g in sub.groupby('noise_strength'):
        detail_path = os.path.join(folder, f'{short}_noise{noise}.csv')
        g[DETAIL_COLS[1:]].to_csv(detail_path, index=False)

    rows = []
    for (noise, grp), g in sub.groupby(['noise_strength', sub.dataset.map(group_of)]):
        r2_raw = pd.to_numeric(g['r2'], errors='coerce')
        r2_clean = r2_raw.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        ok = g[g['status'] == 'ok']
        # 统一复杂度口径：表达式树节点数，从 expr 解析
        comp = pd.Series([unified_complexity(e) for e in ok['expr']],
                         dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        secs = pd.to_numeric(ok['seconds'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            'noise_strength': noise,
            'group': grp,
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
    pd.DataFrame(rows).assign(
        group=lambda d: pd.Categorical(d['group'], categories=['Feynman', 'Strogatz', 'Black-box'], ordered=True)
    ).sort_values(['noise_strength', 'group']).assign(
        group=lambda d: d['group'].astype(str)
    ).to_csv(os.path.join(folder, f'{short}_summary.csv'), index=False)

# 3. 全局 summary（复用 pmlb_results_summary 的口径）
inputs = sorted(glob.glob(args.input_glob))
out_summary = os.path.join(args.results_dir, 'pmlb_results_summary.csv')
subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'pmlb_results_summary.py'),
                '--input_csv', *inputs, '--output_csv', out_summary], check=True)

print(f'\n全量明细: {all_path}（{len(all_df)} 行）')
for algo, short in ALGO_SHORT.items():
    folder = os.path.join(args.results_dir, f'{short}_results')
    files = sorted(os.listdir(folder))
    print(f'{short}_results/: {len(files)} 个文件 -> {", ".join(files)}')
print(f'全局 summary: {out_summary}')
