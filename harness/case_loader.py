from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class CaseLoadError(Exception):
    """Raised when a case file is missing or malformed."""


REQUIRED_FIELDS = [
    "name",
    "category",
    "op",
    "rank_size",
    "dtype",
    "count",
    "scale",
    "backend",
    "enabled",
    "backend_config",
    "expect",
    "notes",
]


def load_case(case_path: str | Path) -> dict[str, Any]:
    path = Path(case_path)

    if not path.exists():
        raise CaseLoadError(f"case file does not exist: {path}")

    if path.suffix not in {".yaml", ".yml"}:
        raise CaseLoadError(f"case file must be .yaml or .yml: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise CaseLoadError(f"failed to parse YAML case {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise CaseLoadError(f"case content must be a YAML mapping: {path}")

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise CaseLoadError(f"case {path} is missing required field(s): {', '.join(missing)}")

    if data["backend"] != "hccl_vm":
        raise CaseLoadError(f"case {path} uses unsupported backend: {data['backend']}")

    if not isinstance(data["backend_config"], dict):
        raise CaseLoadError(f"case {path} field backend_config must be a mapping")

    if "case_config" not in data["backend_config"]:
        raise CaseLoadError(f"case {path} field backend_config.case_config is required")

    if not isinstance(data["expect"], dict):
        raise CaseLoadError(f"case {path} field expect must be a mapping")

    if not isinstance(data["notes"], dict):
        raise CaseLoadError(f"case {path} field notes must be a mapping")

    data["_path"] = str(path)
    return data


def load_cases_from_dir(case_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(case_dir)
    if not root.exists():
        raise CaseLoadError(f"case directory does not exist: {root}")
    if not root.is_dir():
        raise CaseLoadError(f"case path is not a directory: {root}")

    cases: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        cases.append(load_case(path))
    return cases
