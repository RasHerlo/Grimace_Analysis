from __future__ import annotations

from pathlib import Path

import yaml


def repo_root() -> Path:
    """Repo root = the folder holding config.yaml (searched upwards from here)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("config.yaml not found above " + str(Path(__file__)))


def load_config() -> dict:
    root = repo_root()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    data_root = Path(cfg["data_root"])
    cfg["root"] = root
    cfg["data_root"] = data_root
    cfg["overview_path"] = data_root / cfg["overview_xlsx"]
    cfg["baseline_path"] = data_root / cfg["baseline_dir"]
    cfg["stimulated_path"] = data_root / cfg["stimulated_dir"]
    cfg["cache_path"] = root / cfg["cache_dir"]
    return cfg
