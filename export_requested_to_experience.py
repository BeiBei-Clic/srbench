import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import Loader, load

ROOT = Path(__file__).resolve().parent
BLACKBOX_DATASET_DIR = ROOT.parent / "pmlb" / "datasets"
SYMBOLIC_DATASET_DIR = Path("/tmp/pmlb_sym_data")
REQUESTED_RESULTS_DIR = ROOT / "results" / "pmlb_requested"
EXPERIENCE_DIR = ROOT / "results" / "experience"
DATASET_INFO_CSV = ROOT / "docs" / "csv" / "datasets_info.csv"
NOISE_CONFIGS = [
    ("0.0", "noise_0"),
    ("0.001", "noise_0.001"),
    ("0.01", "noise_0.01"),
    ("0.1", "noise_0.1"),
]
GROUP_ORDER = {"Black-box": 0, "Feynman": 1, "Strogatz": 2}


def classify_group(dataset_name):
    if dataset_name.startswith("feynman"):
        return "Feynman"
    if dataset_name.startswith("strogatz"):
        return "Strogatz"
    return "Black-box"


def build_expected_datasets():
    rows = []
    for metadata_path in sorted(BLACKBOX_DATASET_DIR.glob("*/*metadata.yaml")):
        metadata = load(metadata_path.read_text(), Loader=Loader)
        if metadata["task"] != "regression":
            continue
        dataset_name = metadata_path.parent.name
        rows.append(
            {
                "dataset": dataset_name,
                "group": "Black-box",
                "tsv_path": metadata_path.parent / f"{dataset_name}.tsv.gz",
            }
        )

    for dataset_dir in sorted(SYMBOLIC_DATASET_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        rows.append(
            {
                "dataset": dataset_name,
                "group": classify_group(dataset_name),
                "tsv_path": dataset_dir / f"{dataset_name}.tsv.gz",
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["dataset"]).sort_values(
        ["group", "dataset"]
    )
    return df.reset_index(drop=True)


def build_dataset_info(expected_df):
    info_df = pd.read_csv(DATASET_INFO_CSV)
    info_df = info_df.rename(
        columns={"name": "dataset", "nsamples": "n_rows", "nfeatures": "n_cols"}
    )
    info_df["n_features"] = info_df["n_cols"] - 1
    info_df = info_df[["dataset", "n_rows", "n_features"]]
    info_df = info_df.drop_duplicates(subset=["dataset"])
    merged = expected_df.merge(info_df, on="dataset", how="left")
    return merged


def model_complexity(model_str):
    if not isinstance(model_str, str) or model_str == "":
        return np.nan

    from sympy import Abs, Add, Mul, Symbol, cos, exp, log, preorder_traversal, sin, sqrt, tanh
    from sympy.parsing.sympy_parser import parse_expr

    def sub(x, y):
        return Add(x, -y)

    def div(x, y):
        return Mul(x, 1 / y)

    def square(x):
        return x**2

    def cube(x):
        return x**3

    def quart(x):
        return x**4

    local_dict = {
        "add": Add,
        "mul": Mul,
        "sub": sub,
        "div": div,
        "square": square,
        "cube": cube,
        "quart": quart,
        "sqrt": sqrt,
        "log": log,
        "sin": sin,
        "cos": cos,
        "exp": exp,
        "Abs": Abs,
        "Tanh": tanh,
        "Sin": sin,
        "Cos": cos,
        "Log": log,
        "Exp": exp,
        "Sqrt": sqrt,
    }

    for token in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", model_str)):
        if token not in local_dict:
            local_dict[token] = Symbol(token)

    expr = parse_expr(model_str, evaluate=False, local_dict=local_dict)
    count = 0
    for _ in preorder_traversal(expr):
        count += 1
    return count


def export_algorithm(algorithm, dataset_df):
    out_dir = EXPERIENCE_DIR / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for noise_str, noise_dir in NOISE_CONFIGS:
        rows = []
        result_map = {}
        for result_path in sorted(
            (REQUESTED_RESULTS_DIR / noise_dir).glob(f"*/*_{algorithm}_23654.json")
        ):
            result_map[result_path.parent.name] = result_path

        for row in dataset_df.itertuples(index=False):
            dataset_name = row.dataset
            result_path = result_map.get(dataset_name)
            if result_path is None:
                rows.append(
                    {
                        "algorithm": algorithm,
                        "dataset": dataset_name,
                        "status": "error",
                        "n_features": row.n_features,
                        "n_rows": row.n_rows,
                        "r2": np.nan,
                        "rmse": np.nan,
                        "complexity": np.nan,
                        "seconds": np.nan,
                        "error": "missing result",
                        "expression": np.nan,
                    }
                )
                continue

            payload = json.loads(result_path.read_text())
            r2 = payload.get("r2_test", np.nan)
            mse = payload.get("mse_test", np.nan)
            model_str = payload.get("symbolic_model", np.nan)
            status = "ok"
            error = np.nan
            if not np.isfinite(r2):
                status = "error"
                error = "invalid r2"
            rows.append(
                {
                    "algorithm": algorithm,
                    "dataset": dataset_name,
                    "status": status,
                    "n_features": row.n_features,
                    "n_rows": row.n_rows,
                    "r2": r2,
                    "rmse": np.sqrt(mse) if np.isfinite(mse) else np.nan,
                    "complexity": model_complexity(model_str)
                    if status == "ok"
                    else np.nan,
                    "seconds": payload.get("time_time", np.nan),
                    "error": error,
                    "expression": model_str,
                }
            )

        detail_df = pd.DataFrame(rows)
        detail_df = detail_df.sort_values("dataset").reset_index(drop=True)
        detail_path = out_dir / f"{algorithm}_noise{noise_str}.csv"
        detail_df.to_csv(detail_path, index=False)

        detail_df["group"] = detail_df["dataset"].apply(classify_group)
        for group_name, group_df in detail_df.groupby("group", sort=False):
            valid_df = group_df[group_df["r2"].apply(np.isfinite)].copy()
            valid_df["r2_clamped"] = valid_df["r2"].clip(lower=0)

            complexity_vals = valid_df["complexity"].dropna()
            seconds_vals = valid_df["seconds"].dropna()
            valid_count = len(valid_df)
            total_count = len(group_df)
            recovery_count = (valid_df["r2"] >= 0.999).sum()

            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "noise": float(noise_str),
                    "group": group_name,
                    "r2_mean": valid_df["r2_clamped"].mean()
                    if valid_count > 0
                    else np.nan,
                    "r2_std": valid_df["r2_clamped"].std()
                    if valid_count > 0
                    else np.nan,
                    "r2_valid_count": valid_count,
                    "total_count": total_count,
                    "recovery_rate": recovery_count / valid_count
                    if valid_count > 0
                    else np.nan,
                    "complexity_mean": complexity_vals.mean()
                    if len(complexity_vals) > 0
                    else np.nan,
                    "complexity_std": complexity_vals.std()
                    if len(complexity_vals) > 0
                    else np.nan,
                    "complexity_count": len(complexity_vals),
                    "seconds_mean": seconds_vals.mean()
                    if len(seconds_vals) > 0
                    else np.nan,
                    "seconds_std": seconds_vals.std()
                    if len(seconds_vals) > 0
                    else np.nan,
                    "seconds_count": len(seconds_vals),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["group_order"] = summary_df["group"].map(GROUP_ORDER)
    summary_df = summary_df.sort_values(["noise", "group_order"]).drop(
        columns=["group_order"]
    )
    summary_path = out_dir / f"{algorithm}_summary.csv"
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithms", nargs="+")
    args = parser.parse_args()

    expected_df = build_expected_datasets()
    dataset_df = build_dataset_info(expected_df)
    for algorithm in args.algorithms:
        export_algorithm(algorithm, dataset_df)
