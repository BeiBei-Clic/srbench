import numpy as np
import pandas as pd
import jax.numpy as jnp

# Patch: itea 1.0.0 使用了 sklearn 已移除的 squared 参数
import sklearn.metrics
_orig_mse = sklearn.metrics.mean_squared_error
def _patched_mse(*args, **kwargs):
    kwargs.pop('squared', None)
    return _orig_mse(*args, **kwargs)
sklearn.metrics.mean_squared_error = _patched_mse

from itea.regression import ITEA_regressor as _ITEA_regressor
import itea.regression._ITExpr_regressor as _itexpr_mod
_itexpr_mod.mean_squared_error = _patched_mse

# 新版 itea 要求 tfuncs 用 jax.numpy（JAX 自动微分需要）
# 同时手动提供 tfuncs_dx 避免自动生成时的 TracerArrayConversionError
tfuncs = {
    'id': lambda x: x,
    'tanh': jnp.tanh,
    'sin': jnp.sin,
    'cos': jnp.cos,
    'log': jnp.log,
    'exp': jnp.exp,
    'sqrtabs': lambda x: jnp.sqrt(jnp.abs(x)),
}

tfuncs_dx = {
    'id': lambda x: jnp.ones_like(x),
    'tanh': lambda x: 1 - jnp.tanh(x)**2,
    'sin': lambda x: jnp.cos(x),
    'cos': lambda x: -jnp.sin(x),
    'log': lambda x: 1 / x,
    'exp': lambda x: jnp.exp(x),
    'sqrtabs': lambda x: x / (2 * jnp.sqrt(jnp.abs(x)) + 1e-12),
}

hyper_params = [
    {
        "expolim": ((0, 5),),
        "max_terms": ((2, 5),),
        "tfuncs": ({"id": lambda x: x, "sin": jnp.sin},),
        "tfuncs_dx": ({"id": lambda x: jnp.ones_like(x), "sin": lambda x: jnp.cos(x)},),
    },
    {
        "expolim": ((0, 5),),
        "max_terms": ((2, 15),),
        "tfuncs": ({"id": lambda x: x, "sin": jnp.sin},),
        "tfuncs_dx": ({"id": lambda x: jnp.ones_like(x), "sin": lambda x: jnp.cos(x)},),
    },
    {
        "expolim": ((-5, 5),),
        "max_terms": ((2, 15),),
        "tfuncs": ({"id": lambda x: x, "sin": jnp.sin},),
        "tfuncs_dx": ({"id": lambda x: jnp.ones_like(x), "sin": lambda x: jnp.cos(x)},),
    },
    {
        "expolim": ((-5, 5),),
        "max_terms": ((2, 5),),
        "tfuncs": (tfuncs,),
        "tfuncs_dx": (tfuncs_dx,),
    },
    {
        "expolim": ((0, 5),),
        "max_terms": ((2, 15),),
        "tfuncs": (tfuncs,),
        "tfuncs_dx": (tfuncs_dx,),
    },
    {
        "expolim": ((-5, 5),),
        "max_terms": ((2, 15),),
        "tfuncs": (tfuncs,),
        "tfuncs_dx": (tfuncs_dx,),
    },
]

eval_kwargs = {"scale_x": False, "scale_y": False}
est = _ITEA_regressor(
    popsize=1000,
    gens=500,
    expolim=(-1, 1),
    max_terms=2,
    tfuncs=tfuncs,
    tfuncs_dx=tfuncs_dx,
)

# jsonify 会就地修改 get_params() 返回的 dict，把 tfuncs/tfuncs_dx 里的函数转成字符串
# 覆盖 get_params 返回安全副本，避免 est 内部的 tfuncs 被破坏
_orig_get_params = est.get_params
def _safe_get_params(*args, **kwargs):
    params = _orig_get_params(*args, **kwargs)
    # 用字符串占位替换不可序列化的函数值
    for key in ('tfuncs', 'tfuncs_dx'):
        if key in params and isinstance(params[key], dict):
            params[key] = f"<{key} dict>"
    return params
est.get_params = _safe_get_params


def complexity(e):
    return e.bestsol_.n_terms


def model(e, X):
    return e.bestsol_.to_str().replace("^", "**")
