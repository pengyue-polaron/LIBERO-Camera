import os
import sys
from pathlib import Path

import yaml


path = os.path.dirname(os.path.realpath(__file__))
repo_root = Path(path).resolve().parent
sys.path.insert(0, os.path.join(path, "../"))


def _ensure_libero_config():
    config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_file = config_root / "config.yaml"
    if config_file.exists():
        return

    benchmark_root = repo_root / "libero" / "libero"
    datasets_root = Path(
        os.environ.get("LIBERO_DATASETS_PATH", str(repo_root / "datasets"))
    ).expanduser()
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets_root),
        "assets": str(benchmark_root / "assets"),
    }

    config_root.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)


_ensure_libero_config()
