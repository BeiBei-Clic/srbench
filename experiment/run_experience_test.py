"""
批量测试 results/experience 中全部算法在 srbench 评估框架上的可运行性。
用 1089_USCrime 数据集（47行），test=True 小参数模式。
"""
import sys
import os
import json
import time
import traceback
import importlib

# Java 环境（MRGPRegressor 需要）
JAVA_HOME = os.path.expanduser('~/.jdk/jdk-17.0.19+10')
os.environ['JAVA_HOME'] = JAVA_HOME
os.environ['PATH'] = os.path.join(JAVA_HOME, 'bin') + ':' + os.environ.get('PATH', '')

# DSR 需要 Keras 2 兼容模式（TF 2.16+ 默认用 Keras 3，BasicRNNCell 不存在）
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# ellyn 编译时链接了 /tmp/conda-fake/lib 下的 libboost_python310
os.environ['LD_LIBRARY_PATH'] = '/tmp/conda-fake/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# LD_LIBRARY_PATH 进程启动后设置不生效，需要用 ctypes 预加载
import ctypes
ctypes.CDLL('/tmp/conda-fake/lib/libboost_python310.so.1.78.0')

import numpy as np
import pandas as pd

# 将 experiment/ 和 srbench/ 加入 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from method_loader import import_algorithm_module
from evaluate_model import evaluate_model

# results/experience 中的 15 个算法，按 method_loader 能识别的模块名
ALGORITHMS = [
    'AdaBoostRegressor',
    'KernelRidge',
    'MLPRegressor',
    'RandomForestRegressor',
    'gplearn',
    'OperonRegressor',
    'ITEARegressor',
    'MRGPRegressor',
    'LGBMRegressor',
    'XGBRegressor',
    'ffx',
    'gpgomea',
    'tuned.AFPRegressor',
    'tuned.DSRRegressor',
    'tuned.EHCRegressor',
]

DATASET = '/home/xyh/Symbolic_Regression/pmlb/datasets/1089_USCrime/1089_USCrime.tsv.gz'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'experience_test')
SEED = 42


def safe_import(name):
    """安全导入算法模块，失败返回 None 和错误信息。"""
    try:
        return import_algorithm_module(name), ''
    except Exception as e:
        return None, traceback.format_exc()


def safe_evaluate(name, module):
    """安全评估算法，失败返回 None 和错误信息。"""
    try:
        est = module.est
        model_fn = module.model
        eval_kwargs = getattr(module, 'eval_kwargs', {})
        t0 = time.time()
        save_path = evaluate_model(
            dataset=DATASET,
            results_path=RESULTS_DIR,
            random_state=SEED,
            est_name=name,
            est=est,
            model=model_fn,
            test=True,
            **eval_kwargs,
        )
        elapsed = time.time() - t0

        with open(save_path) as f:
            result = json.load(f)

        return result, elapsed, ''
    except Exception:
        return None, 0, traceback.format_exc()


def main():
    print('=' * 70)
    print('srbench 算法批量测试')
    print('数据集:', DATASET)
    print('=' * 70)

    summary = []

    for name in ALGORITHMS:
        print(f'\n{"=" * 60}')
        print(f'测试算法: {name}')
        print(f'{"=" * 60}')

        r2_test = float('nan')
        mse_test = float('nan')
        mae_test = float('nan')
        elapsed = 0.0
        error_msg = ''

        # 阶段 1: 导入
        module, import_err = safe_import(name)
        if module is None:
            print(f'导入失败:\n{import_err}')
            summary.append({
                'algorithm': name,
                'status': 'IMPORT_FAIL',
                'r2_test': r2_test,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'time': '0s',
                'error': import_err.split('\n')[-2] if import_err else '',
            })
            continue

        # 阶段 2: 评估
        result, elapsed, eval_err = safe_evaluate(name, module)
        if result is None:
            print(f'评估失败:\n{eval_err}')
            summary.append({
                'algorithm': name,
                'status': 'EVAL_FAIL',
                'r2_test': r2_test,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'time': '0s',
                'error': eval_err.split('\n')[-2] if eval_err else '',
            })
            continue

        status = 'OK'
        r2_test = result.get('r2_test', float('nan'))
        mse_test = result.get('mse_test', float('nan'))
        mae_test = result.get('mae_test', float('nan'))

        summary.append({
            'algorithm': name,
            'status': status,
            'r2_test': r2_test,
            'mse_test': mse_test,
            'mae_test': mae_test,
            'time': f'{elapsed:.1f}s',
            'error': '',
        })

    # 汇总表格
    print('\n' + '=' * 70)
    print('汇总结果')
    print('=' * 70)
    df = pd.DataFrame(summary)
    if len(df) > 0:
        print(df.to_string(index=False))
        ok_count = (df.status == 'OK').sum()
        fail_count = len(df) - ok_count
        print(f'\n成功: {ok_count}/{len(df)}, 失败: {fail_count}')
    else:
        print('没有算法测试成功')


if __name__ == '__main__':
    main()
