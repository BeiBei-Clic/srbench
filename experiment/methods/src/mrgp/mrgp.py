from sklearn.base import BaseEstimator
import os
import re
import subprocess
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import hashlib


THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def _prefix_expression_to_call(expr):
    tokens = re.findall(r"\(|\)|[^\s()]+", expr)
    op_map = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "mydivide": "div",
        "mylog": "log",
    }

    def parse_node(i):
        token = tokens[i]
        if token != "(":
            return token, i + 1

        op = op_map.get(tokens[i + 1], tokens[i + 1])
        i += 2
        args = []
        while tokens[i] != ")":
            node, i = parse_node(i)
            args.append(node)
        i += 1

        if len(args) == 1:
            return f"{op}({args[0]})", i
        return f"{op}({', '.join(args)})", i

    parsed, end = parse_node(0)
    if end != len(tokens):
        raise ValueError("MRGP expression parse did not consume all tokens")
    return parsed


class MRGPRegressor(BaseEstimator):
    def __init__(
        self,
        g=10,
        popsize=100,
        rt_mut=0.5,
        rt_cross=0.5,
        max_len=10,
        time_out=10 * 60,
        tmp_dir=None,
        n_jobs=1,
        random_state=None,
    ):
        self.g = g
        self.popsize = popsize
        self.rt_cross = rt_cross
        self.rt_mut = rt_mut
        self.max_len = max_len
        self.time_out = time_out
        self.tmp_dir = tmp_dir
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, features, target, sample_weight=None, groups=None):
        data = pd.DataFrame(features)
        data["target"] = target
        row_hashes = hash_pandas_object(data).values
        file_hash = hashlib.sha256(row_hashes).hexdigest()
        data_dir = self.tmp_dir if self.tmp_dir is not None else THIS_DIR
        self.dataset = (
            data_dir
            + "/tmp_data_"
            + file_hash
            + "_"
            + str(np.random.randint(2**15 - 1))
        )
        data.to_csv(self.dataset + "-train", header=None, index=None)
        output = [
            "java",
            "-jar",
            THIS_DIR + "/mrgp.jar",
            "-train",
            self.dataset,
            str(self.g),
            str(self.popsize),
            str(self.rt_mut),
            str(self.rt_cross),
            str(self.max_len),
            str(self.time_out),
            str(self.n_jobs),
        ]
        if self.random_state is not None:
            output.append(str(self.random_state))
        subprocess.check_output(output)
        self.model_, self.complexity_ = self._get_model()
        os.remove(self.dataset + "-train")
        return self

    def predict(self, test, ic=None):
        data = pd.DataFrame(test)
        data["tmp"] = 0
        data.to_csv(self.dataset + "-test", header=None, index=None)
        y_pred = [
            float(x)
            for x in "".join(
                chr(i)
                for i in subprocess.check_output(
                    ["java", "-jar", THIS_DIR + "/mrgp.jar", "-test", self.dataset]
                )
            )[:-1]
            .strip()
            .split(" ")
        ]
        if len(y_pred) != len(test):
            print("ERROR!")
        if np.any(np.isinf(y_pred)):
            print("FOUND INFS!")
        if np.any(np.isnan(y_pred)):
            print("FOUND NANs!")
        os.remove(self.dataset + "-test")
        return y_pred

    def _get_model(self):
        best_data = open(self.dataset + "-best", "r").readline().split(",")
        internal_weights = best_data[2].split(" ")
        model_form = best_data[4].strip()
        complexity_ = 2 + len(internal_weights) * 3
        model_ = _prefix_expression_to_call(model_form)
        return model_, complexity_
