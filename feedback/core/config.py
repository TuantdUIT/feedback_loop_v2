"""Nạp cấu hình YAML của feedback/config và biến môi trường từ .env ở gốc dự án."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "feedback" / "config"


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Đọc ``<name>.yaml``; file rỗng trả dict rỗng."""
    data = yaml.safe_load((config_dir / f"{name}.yaml").read_text(encoding="utf-8"))
    return data or {}


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    path = PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env(key: str, default: str | None = None) -> str | None:
    """Biến môi trường của tiến trình được ưu tiên hơn giá trị trong .env."""
    value = os.environ.get(key)
    if value:
        return value
    return _dotenv().get(key) or default
