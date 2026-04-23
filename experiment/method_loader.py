import importlib
import importlib.util
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT_DIR / "experiment"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


def discover_learners(tuned=False):
    if tuned:
        tuned_dir = EXPERIMENT_DIR / "methods" / "tuned"
        return sorted(
            path.stem
            for path in tuned_dir.glob("*.py")
            if not path.stem.startswith("_")
        )

    learners = {
        path.stem
        for path in (EXPERIMENT_DIR / "methods").glob("*.py")
        if not path.stem.startswith("_")
    }

    learners.update(
        path.name
        for path in (EXPERIMENT_DIR / "methods").iterdir()
        if path.is_dir() and (path / "regressor.py").exists()
    )

    learners.update(
        path.name
        for path in (ROOT_DIR / "algorithms").iterdir()
        if path.is_dir() and (path / "regressor.py").exists()
    )

    return sorted(learners)


def import_algorithm_module(name):
    if name.startswith("tuned."):
        tuned_name = name.split(".", 1)[1]
        tuned_path = EXPERIMENT_DIR / "methods" / "tuned" / f"{tuned_name}.py"
        if not tuned_path.exists():
            raise ModuleNotFoundError(f"找不到 tuned 方法: {name}")
        return importlib.import_module(f"methods.tuned.{tuned_name}")

    candidate_paths = [
        EXPERIMENT_DIR / "methods" / name / "regressor.py",
        EXPERIMENT_DIR / "methods" / f"{name}.py",
        ROOT_DIR / "algorithms" / name / "regressor.py",
    ]

    module_path = None
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            module_path = candidate_path
            break

    if module_path is None:
        raise ModuleNotFoundError(
            "找不到方法模块: {}。检查过: {}".format(
                name,
                ", ".join(str(path) for path in candidate_paths),
            )
        )

    module_name = "srbench_" + name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
