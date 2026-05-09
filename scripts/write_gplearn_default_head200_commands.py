import os
import shlex
from pathlib import Path

from yaml import Loader, load


root = Path(__file__).resolve().parents[1]
data_root = Path("/tmp/pmlb_head200")
results_root = root / "results" / "head200_gplearn_default"
logs_root = root / "results" / "head200_gplearn_default_logs"
command_path = results_root / "run_commands.txt"
seed = 23654
noises = [
    ("0.0", "noise_0"),
    ("0.001", "noise_0.001"),
    ("0.01", "noise_0.01"),
    ("0.1", "noise_0.1"),
]

results_root.mkdir(parents=True, exist_ok=True)
logs_root.mkdir(parents=True, exist_ok=True)
commands = []

for noise_value, noise_dir in noises:
    for metadata_path in sorted(data_root.glob("*/metadata.yaml")):
        metadata = load(metadata_path.read_text(), Loader=Loader)
        if metadata["task"] != "regression":
            continue

        dataset_name = metadata_path.parent.name
        dataset_path = metadata_path.parent / f"{dataset_name}.tsv.gz"
        output_dir = results_root / noise_dir / dataset_name
        output_json = output_dir / f"{dataset_name}_gplearn_{seed}.json"
        log_dir = logs_root / noise_dir
        log_path = log_dir / f"{dataset_name}.log"
        sym_data = dataset_name.startswith("feynman") or dataset_name.startswith("strogatz")

        pythonpath = "/tmp/srbench_vendor"
        if os.environ.get("PYTHONPATH"):
            pythonpath += ":" + os.environ["PYTHONPATH"]

        command_parts = [
            "PYTHONPATH=" + shlex.quote(pythonpath),
            shlex.quote(str(root / ".venv" / "bin" / "python")),
            shlex.quote(str(root / "experiment" / "evaluate_model.py")),
            shlex.quote(str(dataset_path)),
            "-ml",
            "gplearn",
            "-results_path",
            shlex.quote(str(output_dir)),
            "-seed",
            str(seed),
            "-n_jobs",
            "1",
            "-target_noise",
            noise_value,
        ]
        if sym_data:
            command_parts.append("-sym_data")

        commands.append(
            "test -f {output_json} || (mkdir -p {output_dir} {log_dir} && {command} > {log_path} 2>&1)".format(
                output_json=shlex.quote(str(output_json)),
                output_dir=shlex.quote(str(output_dir)),
                log_dir=shlex.quote(str(log_dir)),
                command=" ".join(command_parts),
                log_path=shlex.quote(str(log_path)),
            )
        )

command_path.write_text("\n".join(commands) + "\n")
print("commands", len(commands))
print("saved", command_path)
