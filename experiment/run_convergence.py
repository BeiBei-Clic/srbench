"""
SR 算法收敛曲线实验：7 个算法在 1089_USCrime 数据集上的收敛行为。

gplearn: warm_start 增量训练（细粒度，200 个点）
ITEA: convergence_ 属性（细粒度，500 个点）
Operon, GPGOMEA, AFP, EHC, MRGP: 独立重复运行（8 个 budget checkpoints）

每个独立运行算法通过子进程隔离，避免 C++ 后端内存冲突。
同时采集 training R² 和 test R²。
"""
import sys
import os
import json
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

DATASET = '/home/xyh/Symbolic_Regression/pmlb/datasets/1089_USCrime/1089_USCrime.tsv.gz'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'convergence')
SEED = 42
BUDGETS = [1000, 5000, 10000, 50000, 100000, 200000, 350000, 500000]

# ============================================================
# 数据准备（在主进程中完成，写入临时文件供子进程使用）
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from read_file import read_file

features, labels, feature_names = read_file(DATASET, use_dataframe=True)
X_train_raw_df, X_test_raw_df, y_train_raw, y_test_raw = train_test_split(
    features, labels, train_size=0.75, test_size=0.25, random_state=SEED
)

sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train_raw_df)
X_test_scaled = sc_X.transform(X_test_raw_df)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

# 保存预处理数据供子进程使用
os.makedirs(RESULTS_DIR, exist_ok=True)
data_cache = os.path.join(RESULTS_DIR, '_cache.npz')
np.savez(data_cache,
         X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
         y_train_scaled=y_train_scaled, y_train_raw=y_train_raw,
         y_test_raw=y_test_raw, feature_names=feature_names)
print(f'数据缓存已保存: {data_cache}')

# ============================================================
# 1. gplearn 收敛数据采集（warm_start 增量训练）
# ============================================================
print('\n' + '=' * 60)
print('gplearn 收敛数据采集')
print('=' * 60)

from gplearn.genetic import SymbolicRegressor
from sklearn.utils.validation import validate_data
from sklearn.metrics import r2_score
from types import MethodType

def _vd(self, X, y, y_numeric=True):
    return validate_data(self, X=X, y=y, y_numeric=y_numeric)

est_gp = SymbolicRegressor(
    function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos'),
    population_size=1000, generations=1, warm_start=True, random_state=SEED,
    tournament_size=5, parsimony_coefficient=0.001,
    p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01,
)
est_gp._validate_data = MethodType(_vd, est_gp)

gp_evals = []
gp_r2_test = []
gp_r2_train = []
TOTAL_GENS = 200

for g in range(1, TOTAL_GENS + 1):
    est_gp.generations = g
    est_gp.fit(X_train_scaled_df, y_train_scaled)
    y_pred_test = sc_y.inverse_transform(est_gp.predict(X_test_scaled_df).reshape(-1, 1))
    y_pred_train = sc_y.inverse_transform(est_gp.predict(X_train_scaled_df).reshape(-1, 1))
    r2_test = max(0, r2_score(y_test_raw, y_pred_test))
    r2_train = max(0, r2_score(y_train_raw, y_pred_train))
    gp_evals.append(g * est_gp.population_size)
    gp_r2_test.append(r2_test)
    gp_r2_train.append(r2_train)
    if g % 20 == 0 or g == 1:
        print(f'  gen {g}/{TOTAL_GENS}, evals={gp_evals[-1]}, test R²={r2_test:.4f}, train R²={r2_train:.4f}')

print(f'gplearn 完成: test R²={gp_r2_test[-1]:.4f}, train R²={gp_r2_train[-1]:.4f}')

# ============================================================
# 2. ITEA 收敛数据采集（convergence_ 属性）
# ============================================================
print('\n' + '=' * 60)
print('ITEA 收敛数据采集')
print('=' * 60)

from method_loader import import_algorithm_module
module = import_algorithm_module('ITEARegressor')
est_itea = module.est
est_itea.random_state = SEED
est_itea.fit(X_train_raw_df, y_train_raw)

# 训练 R²（从 convergence_ 的 RMSE 转换）
rmse_list = est_itea.convergence_['fitness']['min']
n_train = len(y_train_raw)
ss_tot = np.sum((y_train_raw - y_train_raw.mean()) ** 2)

itea_evals = []
itea_r2_train = []
for gen, rmse in enumerate(rmse_list, 1):
    mse = rmse ** 2
    r2_train = max(0, 1 - mse * n_train / ss_tot)
    itea_evals.append(gen * est_itea.popsize)
    itea_r2_train.append(r2_train)
    if gen % 100 == 0 or gen == 1:
        print(f'  gen {gen}/{len(rmse_list)}, evals={itea_evals[-1]}, train R²={r2_train:.4f}')

# 最终模型的 test R²
y_pred_test_itea = est_itea.predict(X_test_raw_df)
itea_r2_test_final = max(0, r2_score(y_test_raw, y_pred_test_itea))
print(f'ITEA 完成: 最终 train R²={itea_r2_train[-1]:.4f}, test R²={itea_r2_test_final:.4f}')

# ============================================================
# 3-7. 独立运行算法：通过子进程隔离
# ============================================================

# 子进程脚本模板：读取缓存数据，跑指定算法的 budget checkpoints，写回 JSON
_WORKER_SCRIPT = '''
import sys, os, json, time
sys.path.insert(0, {SCRIPT_DIR!r})
sys.path.insert(0, {ROOT_DIR!r})

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

d = np.load({data_cache!r}, allow_pickle=True)
X_train_scaled = d['X_train_scaled']
X_test_scaled = d['X_test_scaled']
y_train_scaled = d['y_train_scaled']
y_train_raw = d['y_train_raw']
y_test_raw = d['y_test_raw']
feature_names = list(d['feature_names'])

import pandas as pd
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

sc_y = StandardScaler()
sc_y.fit(y_train_raw.reshape(-1, 1))

BUDGETS = {BUDGETS!r}
SEED = {SEED!r}

{ALGO_CODE}

results = dict(evals=evals_list, r2_test=r2_test_list, r2_train=r2_train_list)
with open({output_json!r}, 'w') as f:
    json.dump(results, f)
print(json.dumps(results))
'''

def run_in_subprocess(algo_name, algo_code, timeout=3600):
    """在子进程中运行算法，返回 (evals, r2_test, r2_train) 列表。"""
    output_json = os.path.join(RESULTS_DIR, f'_result_{algo_name}.json')
    script = _WORKER_SCRIPT.format(
        SCRIPT_DIR=SCRIPT_DIR, ROOT_DIR=ROOT_DIR,
        data_cache=data_cache, BUDGETS=BUDGETS, SEED=SEED,
        ALGO_CODE=algo_code, output_json=output_json,
    )
    print(f'\n{"=" * 60}')
    print(f'{algo_name} 收敛数据采集（子进程）')
    print(f'{"=" * 60}')
    proc = subprocess.run(
        [sys.executable, '-u', '-c', script],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        print(f'  子进程错误 (exit {proc.returncode}):')
        for line in proc.stderr.strip().split('\n')[-5:]:
            print(f'    {line}')
        raise RuntimeError(f'{algo_name} 子进程失败 (exit {proc.returncode})')

    # 从 stdout 的最后一行提取 JSON 结果
    lines = proc.stdout.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{'):
            result = json.loads(line)
            evals_list = result['evals']
            r2_test_list = result['r2_test']
            r2_train_list = result['r2_train']
            for i, (e, rt, rn) in enumerate(zip(evals_list, r2_test_list, r2_train_list)):
                print(f'  [{i+1}/{len(evals_list)}] evals={e:>7d}, test R²={rt:.4f}, train R²={rn:.4f}')
            print(f'{algo_name} 完成: 最终 test R²={r2_test_list[-1]:.4f}, train R²={r2_train_list[-1]:.4f}')
            return evals_list, r2_test_list, r2_train_list
    raise RuntimeError(f'{algo_name}: 子进程输出中未找到 JSON 结果')

# --- Operon ---
operon_evals, operon_r2_test, operon_r2_train = run_in_subprocess('Operon', '''
from pyoperon.sklearn import SymbolicRegressor as OperonSR
from sklearn.base import clone

base_operon = OperonSR(
    optimizer_iterations=5, generations=10000, n_threads=1,
    random_state=SEED, population_size=500,
)

evals_list = []
r2_test_list = []
r2_train_list = []
for i, budget in enumerate(BUDGETS):
    t0 = time.time()
    est = clone(base_operon)
    est.max_evaluations = budget
    est.fit(X_train_scaled_df, y_train_scaled)
    y_pred_test = sc_y.inverse_transform(est.predict(X_test_scaled_df).reshape(-1, 1))
    y_pred_train = sc_y.inverse_transform(est.predict(X_train_scaled_df).reshape(-1, 1))
    r2_test = max(0, r2_score(y_test_raw, y_pred_test))
    r2_train = max(0, r2_score(y_train_raw, y_pred_train))
    evals_list.append(budget)
    r2_test_list.append(r2_test)
    r2_train_list.append(r2_train)
    print(f'  [{i+1}/{len(BUDGETS)}] budget={budget:>7d}, test R²={r2_test:.4f}, train R²={r2_train:.4f}, time={time.time()-t0:.1f}s',
          flush=True)
''')

# --- GPGOMEA（每个 checkpoint 独立子进程，避免 C++ 内存累积） ---
_GPG_CHECKPOINT_SCRIPT = '''
import sys, os, json, time, numpy as np, pandas as pd
sys.path.insert(0, {SCRIPT_DIR!r})
sys.path.insert(0, {ROOT_DIR!r})
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

d = np.load({data_cache!r}, allow_pickle=True)
feature_names = list(d['feature_names'])
X_train_scaled_df = pd.DataFrame(d['X_train_scaled'], columns=feature_names)
X_test_scaled_df = pd.DataFrame(d['X_test_scaled'], columns=feature_names)
y_train_scaled = d['y_train_scaled']
y_train_raw = d['y_train_raw']
y_test_raw = d['y_test_raw']

sc_y = StandardScaler()
sc_y.fit(y_train_raw.reshape(-1, 1))

from pygpg.sk import GPGRegressor as GPGR
_orig = GPGR._create_cpp_option_string
def _patched(self):
    orig_get = self.get_params
    def _fp():
        return {{k: v for k, v in orig_get().items() if not k.startswith('_')}}
    self.get_params = _fp
    s = _orig(self)
    self.get_params = orig_get
    return s
GPGR._create_cpp_option_string = _patched

est = GPGR(
    t=2*60*60, g=-1, e={budget}, finetune_max_evals=500, finetune=True,
    tour=4, d=4, pop=512, disable_ims=True,
    feat_sel=20, no_univ_exc_leaves_fos=False, no_large_fos=True,
    bs=2048, fset='+,-,*,/,log,sqrt,sin,cos',
    cmp=0.0, rci=0.0, random_state=42,
)
est.fit(X_train_scaled_df, y_train_scaled)
y_pred_test = sc_y.inverse_transform(np.array(est.predict(X_test_scaled_df)).reshape(-1, 1))
y_pred_train = sc_y.inverse_transform(np.array(est.predict(X_train_scaled_df)).reshape(-1, 1))
r2_test = max(0, r2_score(y_test_raw, y_pred_test))
r2_train = max(0, r2_score(y_train_raw, y_pred_train))
print(r2_test, r2_train)
'''

print(f'\n{"=" * 60}')
print(f'GPGOMEA 收敛数据采集（逐 checkpoint 子进程）')
print(f'{"=" * 60}')

gpg_evals = []
gpg_r2_test = []
gpg_r2_train = []
for i, budget in enumerate(BUDGETS):
    script = _GPG_CHECKPOINT_SCRIPT.format(
        SCRIPT_DIR=SCRIPT_DIR, ROOT_DIR=ROOT_DIR, data_cache=data_cache, budget=budget,
    )
    proc = subprocess.run(
        [sys.executable, '-u', '-c', script],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(f'  [{i+1}/{len(BUDGETS)}] budget={budget:>7d} FAILED (exit {proc.returncode})')
        for line in proc.stderr.strip().split('\n')[-3:]:
            print(f'    {line}')
        gpg_evals.append(budget)
        gpg_r2_test.append(0.0)
        gpg_r2_train.append(0.0)
        continue
    parts = proc.stdout.strip().split('\n')[-1].split()
    r2_test = float(parts[0])
    r2_train = float(parts[1])
    gpg_evals.append(budget)
    gpg_r2_test.append(r2_test)
    gpg_r2_train.append(r2_train)
    print(f'  [{i+1}/{len(BUDGETS)}] budget={budget:>7d}, test R²={r2_test:.4f}, train R²={r2_train:.4f}')

print(f'GPGOMEA 完成: 最终 test R²={gpg_r2_test[-1]:.4f}, train R²={gpg_r2_train[-1]:.4f}')

# --- AFP ---
afp_evals, afp_r2_test, afp_r2_train = run_in_subprocess('AFP', '''
import ctypes
os.environ['LD_LIBRARY_PATH'] = '/tmp/conda-fake/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
ctypes.CDLL('/tmp/conda-fake/lib/libboost_python310.so.1.78.0')

from ellyn import ellyn

evals_list = []
r2_test_list = []
r2_train_list = []
for i, budget in enumerate(BUDGETS):
    g = max(1, budget // 1000)
    actual_evals = g * 1000
    t0 = time.time()
    est = ellyn(
        selection='afp', lex_eps_global=False, lex_eps_dynamic=False,
        islands=False, num_islands=10, island_gens=100,
        verbosity=0, print_data=False, elitism=True, pHC_on=True, prto_arch_on=True,
        max_len=64, max_len_init=20, popsize=1000, g=g,
        time_limit=2*60*60, random_state=SEED,
    )
    est.fit(X_train_scaled, y_train_scaled)
    y_pred_test = sc_y.inverse_transform(est.predict(X_test_scaled).reshape(-1, 1))
    y_pred_train = sc_y.inverse_transform(est.predict(X_train_scaled).reshape(-1, 1))
    r2_test = max(0, r2_score(y_test_raw, y_pred_test))
    r2_train = max(0, r2_score(y_train_raw, y_pred_train))
    evals_list.append(actual_evals)
    r2_test_list.append(r2_test)
    r2_train_list.append(r2_train)
    print(f'  [{i+1}/{len(BUDGETS)}] budget={budget:>7d}, g={g}, evals={actual_evals:>7d}, '
          f'test R²={r2_test:.4f}, train R²={r2_train:.4f}, time={time.time()-t0:.1f}s', flush=True)
''')

# --- EHC ---
ehc_evals, ehc_r2_test, ehc_r2_train = run_in_subprocess('EHC', '''
import ctypes
os.environ['LD_LIBRARY_PATH'] = '/tmp/conda-fake/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
ctypes.CDLL('/tmp/conda-fake/lib/libboost_python310.so.1.78.0')

from ellyn import ellyn

evals_list = []
r2_test_list = []
r2_train_list = []
for i, budget in enumerate(BUDGETS):
    g = max(1, budget // 4000)
    actual_evals = g * 4000
    t0 = time.time()
    est = ellyn(
        eHC_on=True, eHC_its=3, selection='afp',
        lex_eps_global=False, lex_eps_dynamic=False,
        islands=False, num_islands=10, island_gens=100,
        verbosity=0, print_data=False, elitism=True, pHC_on=True, prto_arch_on=True,
        max_len=64, max_len_init=20, popsize=1000, g=g,
        time_limit=2*60*60, random_state=SEED,
    )
    est.fit(X_train_scaled, y_train_scaled)
    y_pred_test = sc_y.inverse_transform(est.predict(X_test_scaled).reshape(-1, 1))
    y_pred_train = sc_y.inverse_transform(est.predict(X_train_scaled).reshape(-1, 1))
    r2_test = max(0, r2_score(y_test_raw, y_pred_test))
    r2_train = max(0, r2_score(y_train_raw, y_pred_train))
    evals_list.append(actual_evals)
    r2_test_list.append(r2_test)
    r2_train_list.append(r2_train)
    print(f'  [{i+1}/{len(BUDGETS)}] budget={budget:>7d}, g={g}, evals={actual_evals:>7d}, '
          f'test R²={r2_test:.4f}, train R²={r2_train:.4f}, time={time.time()-t0:.1f}s', flush=True)
''')

# --- MRGP（限制到前 4 个 budget，避免 Java 子进程超时） ---
mrgp_evals, mrgp_r2_test, mrgp_r2_train = run_in_subprocess('MRGP', '''
os.environ['JAVA_HOME'] = os.path.expanduser('~/.jdk/jdk-17.0.19+10')
os.environ['PATH'] = os.path.join(os.environ['JAVA_HOME'], 'bin') + ':' + os.environ.get('PATH', '')

from methods.src.mrgp import MRGPRegressor as _MRGPRegressor

# MRGP 默认 g=10, popsize=100，只跑前 4 个 budget（最多 50000 evals = g=500）
MRGP_BUDGETS = BUDGETS[:4]

evals_list = []
r2_test_list = []
r2_train_list = []
for i, budget in enumerate(MRGP_BUDGETS):
    g = max(1, budget // 100)
    actual_evals = g * 100
    t0 = time.time()
    est = _MRGPRegressor(g=g, popsize=100, max_len=6, random_state=SEED, time_out=2*60*60)
    est.fit(X_train_scaled_df, y_train_scaled)
    y_pred_test_arr = np.array(est.predict(X_test_scaled_df))
    y_pred_train_arr = np.array(est.predict(X_train_scaled_df))
    y_pred_test = sc_y.inverse_transform(y_pred_test_arr.reshape(-1, 1))
    y_pred_train = sc_y.inverse_transform(y_pred_train_arr.reshape(-1, 1))
    r2_test = max(0, r2_score(y_test_raw, y_pred_test))
    r2_train = max(0, r2_score(y_train_raw, y_pred_train))
    evals_list.append(actual_evals)
    r2_test_list.append(r2_test)
    r2_train_list.append(r2_train)
    print(f'  [{i+1}/{len(MRGP_BUDGETS)}] budget={budget:>7d}, g={g}, evals={actual_evals:>7d}, '
          f'test R²={r2_test:.4f}, train R²={r2_train:.4f}, time={time.time()-t0:.1f}s', flush=True)
''', timeout=3600)

# ============================================================
# 8. 保存 JSON
# ============================================================
convergence_data = {
    'gplearn': {'evals': gp_evals, 'r2_test': gp_r2_test, 'r2_train': gp_r2_train},
    'ITEA': {'evals': itea_evals, 'r2_train': itea_r2_train, 'r2_test_final': itea_r2_test_final},
    'Operon': {'evals': operon_evals, 'r2_test': operon_r2_test, 'r2_train': operon_r2_train},
    'GPGOMEA': {'evals': gpg_evals, 'r2_test': gpg_r2_test, 'r2_train': gpg_r2_train},
    'AFP': {'evals': afp_evals, 'r2_test': afp_r2_test, 'r2_train': afp_r2_train},
    'EHC': {'evals': ehc_evals, 'r2_test': ehc_r2_test, 'r2_train': ehc_r2_train},
    'MRGP': {'evals': mrgp_evals, 'r2_test': mrgp_r2_test, 'r2_train': mrgp_r2_train},
}

json_path = os.path.join(RESULTS_DIR, 'convergence_data.json')
with open(json_path, 'w') as f:
    json.dump(convergence_data, f, indent=2)
print(f'\nJSON 已保存: {json_path}')

# 清理临时文件
for fname in ['_cache.npz', '_result_Operon.json', '_result_GPGOMEA.json',
              '_result_AFP.json', '_result_EHC.json', '_result_MRGP.json']:
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)

# ============================================================
# 9. 绘制收敛曲线（左图=train R²，右图=test R²）
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

all_curves = [
    ('gplearn', gp_evals, gp_r2_test, gp_r2_train, None),
    ('ITEA', itea_evals, None, itea_r2_train, None),
    ('Operon', operon_evals, operon_r2_test, operon_r2_train, 'o'),
    ('GPGOMEA', gpg_evals, gpg_r2_test, gpg_r2_train, 's'),
    ('AFP', afp_evals, afp_r2_test, afp_r2_train, '^'),
    ('EHC', ehc_evals, ehc_r2_test, ehc_r2_train, 'D'),
    ('MRGP', mrgp_evals, mrgp_r2_test, mrgp_r2_train, 'v'),
]

for idx, (name, evals, r2_test, r2_train, marker) in enumerate(all_curves):
    c = colors[idx]
    kw = dict(linewidth=1.5, color=c)
    if marker:
        kw['marker'] = marker
        kw['markersize'] = 4
    # train R² (左图)
    ax_train.plot(evals, r2_train, label=name, **kw)
    # test R² (右图)
    if r2_test is not None:
        ax_test.plot(evals, r2_test, label=name, **kw)
    elif name == 'ITEA':
        ax_test.plot(itea_evals[-1], itea_r2_test_final, '*', markersize=10, color=c,
                     label=f'ITEA ({itea_r2_test_final:.3f})')

for ax, title in [(ax_train, 'Training R²'), (ax_test, 'Test R²')]:
    ax.set_xscale('log')
    ax.set_xlabel('Evaluations', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

plt.tight_layout()

pdf_path = os.path.join(RESULTS_DIR, 'convergence_curves.pdf')
fig.savefig(pdf_path, format='pdf')
plt.close(fig)
print(f'PDF 已保存: {pdf_path}')
