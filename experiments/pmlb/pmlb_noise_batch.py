"""
批量噪声实验：本地 pmlb/datasets 全部回归数据集 x 4 个算法 x 多档噪声。

每个 (数据集, 算法, 噪声档) 组合起一个子进程跑 pmlb_noise_single.py，
默认只取每个数据集前 200 行。每个噪声档一份结果 CSV：
  {基础名}_noise{强度}.csv（noise 0 也带 _noise0.0 后缀）
断点续跑：已有 CSV 中出现过的 (dataset, algorithm) 直接跳过，只追加不覆盖。
单组合超过 3900 秒强制终止记 status=timeout；子进程失败记 status=error，
不影响整批。

smoke:
  .venv/bin/python experiments/pmlb/pmlb_noise_batch.py --dataset_limit 2 \
      --algorithms ffx,tuned.OperonRegressor --noises 0.1 --max_rows 100 --jobs 4
全量（默认参数）:
  .venv/bin/python experiments/pmlb/pmlb_noise_batch.py
"""
import sys
import os
import csv
import json
import glob
import time
import argparse
import subprocess
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PYTHON = os.path.join(ROOT_DIR, '.venv', 'bin', 'python')
SINGLE = os.path.join(SCRIPT_DIR, 'pmlb_noise_single.py')

parser = argparse.ArgumentParser(description='pmlb 批量噪声实验')
parser.add_argument('--datasets_dir', type=str, default='/home/xyh/Symbolic_Regression/pmlb/datasets')
parser.add_argument('--algorithms', type=str,
                    default='tuned.OperonRegressor,tuned.MRGPRegressor,tuned.ITEARegressor,ffx')
parser.add_argument('--noises', type=str, default='0,0.001,0.01,0.1',
                    help='逗号分隔的乘性噪声强度列表')
parser.add_argument('--noise_seed', type=int, default=0)
parser.add_argument('--max_rows', type=int, default=200, help='每个数据集只取前 N 行')
parser.add_argument('--seed', type=int, default=42, help='train/test 划分种子')
parser.add_argument('--jobs', type=int, default=56)
parser.add_argument('--dataset_limit', type=int, default=0, help='只跑按名称排序的前 N 个数据集（冒烟用）')
parser.add_argument('--task_timeout', type=int, default=3900, help='单组合墙钟上限（秒）')
parser.add_argument('--output_csv', type=str, default='pmlb_results_srbench',
                    help='结果 CSV 基础名，实际文件自动补 _noise{强度}.csv 后缀')
parser.add_argument('--results_dir', type=str, default=os.path.join(SCRIPT_DIR, 'results'))
args = parser.parse_args()

ALGORITHMS = args.algorithms.split(',')
NOISES = [float(n) for n in args.noises.split(',')]
FIELDS = ['dataset', 'algorithm', 'status', 'n_features', 'r2', 'rmse',
          'complexity', 'seconds', 'error', 'expr']
TMP_DIR = os.path.join(args.results_dir, '_tmp')
os.makedirs(args.results_dir, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def discover_datasets():
    """全部回归数据集（metadata.yaml 过滤分类集），按名称排序保证稳定序号。"""
    names = []
    for meta in sorted(glob.glob(args.datasets_dir + '/*/metadata.yaml')):
        name = meta.split('/')[-2]
        with open(meta) as f:
            if 'task: regression' in f.read():
                names.append(name)
    return names


def csv_path(noise):
    return os.path.join(args.results_dir, f'{args.output_csv}_noise{noise}.csv')


def run_task(task):
    """子进程跑单组合，返回 (noise, 结果行 dict)。"""
    idx, name, dpath, algo, noise = task
    result_json = os.path.join(TMP_DIR, f'{name}_{algo.replace(".", "_")}_noise{noise}.json')
    cmd = [
        PYTHON, SINGLE,
        '--dataset_path', dpath, '--algorithm', algo,
        '--noise_strength', str(noise), '--noise_seed', str(args.noise_seed),
        '--dataset_index', str(idx), '--max_rows', str(args.max_rows),
        '--seed', str(args.seed), '--result_json', result_json,
    ]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env=env, text=True)
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < args.task_timeout:
        time.sleep(5)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
        row = {'dataset': name, 'algorithm': algo.replace('tuned.', ''), 'status': 'timeout',
               'n_features': '', 'r2': '', 'rmse': '', 'complexity': '', 'seconds': '',
               'error': f'wall clock > {args.task_timeout}s', 'expr': ''}
        return noise, row
    if proc.returncode != 0:
        stderr = (proc.stderr.read() or '').strip().split('\n')
        row = {'dataset': name, 'algorithm': algo.replace('tuned.', ''), 'status': 'error',
               'n_features': '', 'r2': '', 'rmse': '', 'complexity': '', 'seconds': '',
               'error': stderr[-1][:300] if stderr else f'exit {proc.returncode}', 'expr': ''}
        return noise, row
    with open(result_json) as f:
        row = json.load(f)
    return noise, row


if __name__ == '__main__':
    names = discover_datasets()
    if args.dataset_limit:
        names = names[:args.dataset_limit]
    print(f'数据集 {len(names)} 个，算法 {ALGORITHMS}，噪声档 {NOISES}，'
          f'max_rows={args.max_rows}，并行 {args.jobs}')

    # 断点续跑：各噪声档已出现的 (dataset, algorithm) 跳过，只追加不覆盖
    done = set()
    files, writers = {}, {}
    for noise in NOISES:
        path = csv_path(noise)
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    done.add((noise, row['dataset'], row['algorithm']))
        fresh = not os.path.exists(path)
        files[noise] = open(path, 'a', newline='')
        writers[noise] = csv.DictWriter(files[noise], fieldnames=FIELDS)
        if fresh:
            writers[noise].writeheader()

    tasks = []
    for idx, name in enumerate(names):
        dpath = os.path.join(args.datasets_dir, name, f'{name}.tsv.gz')
        for algo in ALGORITHMS:
            for noise in NOISES:
                if (noise, name, algo.replace('tuned.', '')) in done:
                    continue
                tasks.append((idx, name, dpath, algo, noise))
    print(f'待跑 {len(tasks)} 个组合（断点续跑已跳过 {len(done)} 条已有结果）')
    if not tasks:
        sys.exit(0)

    n_ok = n_fail = 0
    with Pool(args.jobs) as pool:
        for i, (noise, row) in enumerate(pool.imap_unordered(run_task, tasks)):
            writers[noise].writerow(row)
            files[noise].flush()
            if row['status'] == 'ok':
                n_ok += 1
            else:
                n_fail += 1
            print(f"[{i+1}/{len(tasks)}] {row['status']:>7s} {row['dataset']} | "
                  f"{row['algorithm']} | noise={noise}", flush=True)
    for f in files.values():
        f.close()

    print(f'\n完成 ok={n_ok}，失败/超时={n_fail}，共 {len(tasks)}')
    for noise in NOISES:
        print(f"  noise={noise}: {csv_path(noise)}")
