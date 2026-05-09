import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import Loader, load


root = Path(__file__).resolve().parents[1]
dataset_root = root.parent / "pmlb" / "datasets"
raw_root = root / "results" / "head200_gplearn_default"
out_dir = root / "results" / "experience" / "gplearn"
if len(sys.argv) == 2:
    raw_root = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

noise_configs = [
    ("0.0", "noise_0"),
    ("0.001", "noise_0.001"),
    ("0.01", "noise_0.01"),
    ("0.1", "noise_0.1"),
]
group_order = {"Black-box": 0, "Feynman": 1, "Strogatz": 2}

dataset_rows = []
for metadata_path in sorted(dataset_root.glob("*/*metadata.yaml")):
    metadata = load(metadata_path.read_text(), Loader=Loader)
    if metadata["task"] != "regression":
        continue
    dataset_name = metadata_path.parent.name
    group = "Black-box"
    if dataset_name.startswith("feynman"):
        group = "Feynman"
    if dataset_name.startswith("strogatz"):
        group = "Strogatz"
    dataset_rows.append({"dataset": dataset_name, "group": group})

dataset_df = pd.DataFrame(dataset_rows).sort_values("dataset").reset_index(drop=True)
summary_rows = []

for noise_str, noise_dir in noise_configs:
    detail_rows = []
    for dataset_row in dataset_df.itertuples(index=False):
        dataset_name = dataset_row.dataset
        result_path = raw_root / noise_dir / dataset_name / f"{dataset_name}_gplearn_23654.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            r2 = payload["r2_test"]
            mse = payload["mse_test"]
            expression = payload["symbolic_model"]
            status = "ok"
            error = ""
            if not np.isfinite(r2):
                status = "error"
                error = "invalid r2"
            complexity = len(re.split(r"\(|,", expression)) if status == "ok" else np.nan
            seconds = payload["time_time"] if status == "ok" else np.nan
            rmse = np.sqrt(mse) if np.isfinite(mse) else np.nan
        else:
            status = "error"
            r2 = np.nan
            rmse = np.nan
            complexity = np.nan
            seconds = np.nan
            error = "missing result"
            expression = np.nan

        detail_rows.append(
            {
                "algorithm": "gplearn",
                "dataset": dataset_name,
                "status": status,
                "r2": r2,
                "rmse": rmse,
                "complexity": complexity,
                "seconds": seconds,
                "error": error,
                "expression": expression,
                "group": dataset_row.group,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.drop(columns=["group"]).to_csv(out_dir / f"gplearn_noise{noise_str}.csv", index=False)

    for group_name, group_df in detail_df.groupby("group", sort=False):
        r2_raw = pd.to_numeric(group_df["r2"], errors="coerce")
        r2_for_stats = r2_raw.where(np.isfinite(r2_raw), 0).clip(lower=0)
        r2_valid_count = int((np.isfinite(r2_raw) & (r2_raw >= 0)).sum())
        total_count = len(group_df)
        recovery_rate = float((r2_for_stats > 0.9).sum() / total_count)

        ok_df = group_df[group_df["status"] == "ok"]
        complexity_vals = pd.to_numeric(ok_df["complexity"], errors="coerce")
        complexity_vals = complexity_vals[np.isfinite(complexity_vals)]
        seconds_vals = pd.to_numeric(ok_df["seconds"], errors="coerce")
        seconds_vals = seconds_vals[np.isfinite(seconds_vals)]

        summary_rows.append(
            {
                "algorithm": "gplearn",
                "noise": float(noise_str),
                "group": group_name,
                "r2_mean": r2_for_stats.mean(),
                "r2_std": r2_for_stats.std(ddof=0),
                "r2_valid_count": r2_valid_count,
                "total_count": total_count,
                "recovery_rate": recovery_rate,
                "complexity_mean": complexity_vals.mean() if len(complexity_vals) else np.nan,
                "complexity_std": complexity_vals.std(ddof=0) if len(complexity_vals) else np.nan,
                "complexity_count": len(complexity_vals),
                "seconds_mean": seconds_vals.mean() if len(seconds_vals) else np.nan,
                "seconds_std": seconds_vals.std(ddof=0) if len(seconds_vals) else np.nan,
                "seconds_count": len(seconds_vals),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_df["group_order"] = summary_df["group"].map(group_order)
summary_df = summary_df.sort_values(["noise", "group_order"]).drop(columns=["group_order"])
summary_df.to_csv(out_dir / "gplearn_summary.csv", index=False)

print(summary_df.to_string(index=False))
print("saved", out_dir / "gplearn_summary.csv")
