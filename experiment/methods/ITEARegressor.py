import os
import sys
import numpy as np
import pandas as pd

prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
os.environ["LD_LIBRARY_PATH"] = prefix + "/lib"
os.environ["PATH"] = prefix + "/bin:" + os.environ.get("PATH", "")

import pyITEA as itea


hyper_params = [
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 5),),
        "transfunctions": ("[Id, Sin]",),
    },
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Sin]",),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Sin]",),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 5),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
    },
    {
        "exponents": ((0, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
    },
    {
        "exponents": ((-5, 5),),
        "termlimit": ((2, 15),),
        "transfunctions": ("[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",),
    },
]

eval_kwargs = {"scale_x": False, "scale_y": False}
est = itea.ITEARegressor(
    npop=1000,
    ngens=500,
    exponents=(-1, 1),
    termlimit=(2, 2),
    nonzeroexps=1,
    transfunctions="[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]",
)
_original_predict = est.predict


def _predict_with_float_inputs(X):
    if isinstance(X, pd.DataFrame):
        X = X.astype(float)
    else:
        X = np.asarray(X, dtype=float)
    return _original_predict(X)


est.predict = _predict_with_float_inputs


def complexity(e):
    return e.len


def model(e, X):
    new_model = e.sympy.replace("^", "**")
    for i, f in reversed(list(enumerate(X.columns))):
        new_model = new_model.replace(f"x{i}", f)
    return new_model
