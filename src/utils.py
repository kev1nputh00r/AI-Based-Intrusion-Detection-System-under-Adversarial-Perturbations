from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Paths:
    artifacts_dir: str
    reports_dir: str
    models_dir: str
    metrics_file: str


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.artifacts_dir, exist_ok=True)
    os.makedirs(paths.reports_dir, exist_ok=True)
    os.makedirs(paths.models_dir, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
