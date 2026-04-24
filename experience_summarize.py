import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/experience")


def classify_group(dataset_name):
    if dataset_name.startswith("feynman"):
        return "Feynman"
    elif dataset_name.startswith("strogatz"):
        return "Strogatz"
    else:
        return "Black-box"


def summarize_algorithm(algo_dir):
    algo_name = algo_dir.name
    csv_files = sorted(algo_dir.glob(f"{algo_name}_noise*.csv"))
    if not csv_files:
        print(f"  跳过 {algo_name}，无噪声 CSV 文件")
        return

    rows = []
    for csv_file in csv_files:
        noise_str = csv_file.stem.split("noise")[1]
        noise = float(noise_str)
        df = pd.read_csv(csv_file)

        # 只统计 status==ok 的行
        df_ok = df[df["status"] == "ok"].copy()
        if df_ok.empty:
            print(f"  {csv_file.name}: 无 ok 结果")
            continue

        # r2 < 0 的按 0 计算
        df_ok["r2_clamped"] = df_ok["r2"].clip(lower=0)

        df_ok["group"] = df_ok["dataset"].apply(classify_group)

        for group_name, group_df in df_ok.groupby("group"):
            r2_vals = group_df["r2_clamped"]
            total = len(group_df)

            r2_mean = r2_vals.mean()
            r2_std = r2_vals.std()

            # recovery rate: r2 >= 0.999 视为恢复
            recovery_count = (group_df["r2"] >= 0.999).sum()
            recovery_rate = recovery_count / total

            # complexity 统计（非空值）
            complexity_vals = group_df["complexity"].dropna()
            complexity_mean = complexity_vals.mean() if len(complexity_vals) > 0 else np.nan
            complexity_std = complexity_vals.std() if len(complexity_vals) > 0 else np.nan
            complexity_count = len(complexity_vals)

            # seconds 统计（非空值）
            seconds_vals = group_df["seconds"].dropna()
            seconds_mean = seconds_vals.mean() if len(seconds_vals) > 0 else np.nan
            seconds_std = seconds_vals.std() if len(seconds_vals) > 0 else np.nan
            seconds_count = len(seconds_vals)

            rows.append({
                "algorithm": algo_name,
                "noise": noise,
                "group": group_name,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "r2_valid_count": total,
                "total_count": total,
                "recovery_rate": recovery_rate,
                "complexity_mean": complexity_mean,
                "complexity_std": complexity_std,
                "complexity_count": complexity_count,
                "seconds_mean": seconds_mean,
                "seconds_std": seconds_std,
                "seconds_count": seconds_count,
            })

    if not rows:
        print(f"  {algo_name}: 无有效数据")
        return

    summary_df = pd.DataFrame(rows)
    out_path = algo_dir / f"{algo_name}_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"  {algo_name}: {len(rows)} 行 -> {out_path}")


def main():
    for algo_dir in sorted(RESULTS_DIR.iterdir()):
        if not algo_dir.is_dir():
            continue
        summarize_algorithm(algo_dir)
    print("完成")


if __name__ == "__main__":
    main()
