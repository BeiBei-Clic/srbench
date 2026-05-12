"""
SR 算法收敛曲线实验：gplearn (test R²) 和 ITEA (train R²)。
用 1089_USCrime 数据集，默认配置跑训练，记录 R² 随评估次数的变化。
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from read_file import read_file
from method_loader import import_algorithm_module

DATASET = '/home/xyh/Symbolic_Regression/pmlb/datasets/1089_USCrime/1089_USCrime.tsv.gz'
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'convergence')
SEED = 42

# ============================================================
# 数据加载与划分（内联自 evaluate_model.py 逻辑）
# ============================================================
features, labels, feature_names = read_file(DATASET, use_dataframe=True)
X_train_raw_df, X_test_raw_df, y_train_raw, y_test_raw = train_test_split(
    features, labels, train_size=0.75, test_size=0.25, random_state=SEED
)

# gplearn 用 scaled 数据（DataFrame）
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train_raw_df)
X_test_scaled = sc_X.transform(X_test_raw_df)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

# ============================================================
# 1. gplearn 收敛数据采集
# ============================================================
print('=' * 60)
print('gplearn 收敛数据采集')
print('=' * 60)

from gplearn.genetic import SymbolicRegressor
from sklearn.utils.validation import validate_data
from types import MethodType

def _vd(self, X, y, y_numeric=True):
    return validate_data(self, X=X, y=y, y_numeric=y_numeric)

est_gp = SymbolicRegressor(
    function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos'),
    population_size=1000,
    generations=1,
    warm_start=True,
    random_state=SEED,
    tournament_size=5,
    parsimony_coefficient=0.001,
    p_crossover=0.9,
    p_subtree_mutation=0.01,
    p_hoist_mutation=0.01,
    p_point_mutation=0.01,
)
est_gp._validate_data = MethodType(_vd, est_gp)

gp_evals = []
gp_r2 = []
TOTAL_GENS = 200

for g in range(1, TOTAL_GENS + 1):
    est_gp.generations = g
    est_gp.fit(X_train_scaled_df, y_train_scaled)
    y_pred = sc_y.inverse_transform(
        est_gp.predict(X_test_scaled_df).reshape(-1, 1)
    )
    r2 = max(0, r2_score(y_test_raw, y_pred))
    gp_evals.append(g * est_gp.population_size)
    gp_r2.append(r2)
    if g % 20 == 0 or g == 1:
        print(f'  gen {g}/{TOTAL_GENS}, evals={gp_evals[-1]}, R²={r2:.4f}')

print(f'gplearn 完成: 最终 R²={gp_r2[-1]:.4f}')

# ============================================================
# 2. ITEA 收敛数据采集
# ============================================================
print('\n' + '=' * 60)
print('ITEA 收敛数据采集')
print('=' * 60)

module = import_algorithm_module('ITEARegressor')
est_itea = module.est
est_itea.random_state = SEED
est_itea.fit(X_train_raw_df, y_train_raw)

rmse_list = est_itea.convergence_['fitness']['min']
n_train = len(y_train_raw)
ss_tot = np.sum((y_train_raw - y_train_raw.mean()) ** 2)

itea_evals = []
itea_r2 = []
for gen, rmse in enumerate(rmse_list, 1):
    mse = rmse ** 2
    r2_train = max(0, 1 - mse * n_train / ss_tot)
    itea_evals.append(gen * est_itea.popsize)
    itea_r2.append(r2_train)
    if gen % 100 == 0 or gen == 1:
        print(f'  gen {gen}/{len(rmse_list)}, evals={itea_evals[-1]}, R²={r2_train:.4f}')

print(f'ITEA 完成: 最终 R²={itea_r2[-1]:.4f}')

# ============================================================
# 3. 保存 JSON
# ============================================================
os.makedirs(RESULTS_DIR, exist_ok=True)

convergence_data = {
    'gplearn': {'evals': gp_evals, 'r2': gp_r2, 'label': 'gplearn (test R²)'},
    'ITEA': {'evals': itea_evals, 'r2': itea_r2, 'label': 'ITEA (train R²)'},
}

json_path = os.path.join(RESULTS_DIR, 'convergence_data.json')
with open(json_path, 'w') as f:
    json.dump(convergence_data, f, indent=2)
print(f'\nJSON 已保存: {json_path}')

# ============================================================
# 4. 绘制 SVG
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(gp_evals, gp_r2, label='gplearn (test R²)', linewidth=1.5)
ax.plot(itea_evals, itea_r2, label='ITEA (train R²)', linewidth=1.5)

ax.set_xscale('log')
ax.set_xlabel('Evaluations', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('SR Algorithm Convergence Curves (1089_USCrime)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

plt.tight_layout()

svg_path = os.path.join(RESULTS_DIR, 'convergence_curves.svg')
fig.savefig(svg_path, format='svg')
plt.close(fig)
print(f'SVG 已保存: {svg_path}')
