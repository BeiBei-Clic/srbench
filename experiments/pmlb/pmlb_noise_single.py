"""
单个 (数据集, 算法, 噪声档) 组合的训练评估 runner，供 pmlb_noise_batch.py 子进程调用。

流程：读数据取前 max_rows 行 -> 75/25 划分 -> 训练集目标加乘性高斯噪声
y*(1+xi), xi~N(0,noise_strength^2) -> 按 eval_kwargs 标准化 -> fit ->
原始尺度干净测试集上算 r2/rmse -> 结果 JSON 写入 --result_json。

噪声种子 = noise_seed + dataset_index（dataset_index 基于批量脚本中完整
数据集列表的稳定顺序传入），noise_strength == 0 时直接用原始 y。
"""
import sys
import os
import json
import time
import argparse
import inspect

# Java 环境（MRGPRegressor 需要）
JAVA_HOME = os.path.expanduser('~/.jdk/jdk-17.0.19+10')
os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['PATH'] = os.path.join(JAVA_HOME, 'bin') + ':' + os.environ.get('PATH', '')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # srbench 仓库根
EXPERIMENT_DIR = os.path.join(ROOT_DIR, 'experiment')
sys.path.insert(0, EXPERIMENT_DIR)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from read_file import read_file
from method_loader import import_algorithm_module

parser = argparse.ArgumentParser(description='单个数据集-算法-噪声组合的训练评估')
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--algorithm', type=str, required=True,
                    help='算法模块名，同 method_loader.import_algorithm_module')
parser.add_argument('--noise_strength', type=float, required=True)
parser.add_argument('--noise_seed', type=int, default=0)
parser.add_argument('--dataset_index', type=int, required=True,
                    help='数据集在完整数据集列表中的稳定序号，用于噪声种子')
parser.add_argument('--max_rows', type=int, default=200)
parser.add_argument('--seed', type=int, default=42, help='train/test 划分种子')
parser.add_argument('--result_json', type=str, required=True)
args = parser.parse_args()

t0 = time.time()

# 数据加载，只取前 max_rows 行
X, y, feature_names = read_file(args.dataset_path, use_dataframe=True)
X = X.iloc[:args.max_rows]
y = y[:args.max_rows]
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=0.25, random_state=args.seed)

# 乘性高斯噪声：只加训练集目标，测试集保持干净
if args.noise_strength > 0:
    rng = np.random.RandomState(args.noise_seed + args.dataset_index)
    y_train = y_train * (1.0 + rng.normal(0, args.noise_strength, size=len(y_train)))

# 每个组合重新导入模块拿全新 est（est 直接 fit，不 clone）
module = import_algorithm_module(args.algorithm)
est = module.est
if hasattr(est, 'random_state'):
    est.random_state = args.seed
eval_kwargs = getattr(module, 'eval_kwargs', {})
scale_x = eval_kwargs.get('scale_x', True)
scale_y = eval_kwargs.get('scale_y', True)

sc_X = StandardScaler()
if scale_x:
    X_train_fit = sc_X.fit_transform(X_train)
    X_test_fit = sc_X.transform(X_test)
else:
    X_train_fit, X_test_fit = X_train, X_test
import pandas as pd
if isinstance(X_train_fit, np.ndarray):
    X_train_fit = pd.DataFrame(X_train_fit, columns=feature_names)
    X_test_fit = pd.DataFrame(X_test_fit, columns=feature_names)

sc_y = StandardScaler()
if scale_y:
    y_train_fit = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()
else:
    y_train_fit = y_train.astype(float)

t_fit = time.time()
est.fit(X_train_fit, y_train_fit)
fit_seconds = time.time() - t_fit

# 原始尺度干净测试集上评估
y_pred = np.asarray(est.predict(X_test_fit), dtype=float).reshape(-1, 1)
if scale_y:
    y_pred = sc_y.inverse_transform(y_pred)
y_pred = y_pred.flatten()

complexity = module.complexity(est)
if 'X' in inspect.signature(module.model).parameters:
    expr = module.model(est, X_train_fit)
else:
    expr = module.model(est)

result = {
    'dataset': args.dataset_path.split('/')[-1].split('.tsv.gz')[0],
    'algorithm': args.algorithm.replace('tuned.', ''),
    'status': 'ok',
    'n_features': n_features,
    'r2': float(r2_score(y_test, y_pred)),
    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
    'complexity': int(complexity) if complexity is not None else None,
    'seconds': round(time.time() - t0, 1),
    'error': '',
    'expr': str(expr),
}
with open(args.result_json, 'w') as f:
    json.dump(result, f)
print('done', result['dataset'], result['algorithm'], args.noise_strength)
