"""Configuración centralizada y resolución de rutas.

Todas las rutas de config.yaml son relativas a la raíz del proyecto y se resuelven aquí,
de modo que el código se comporta igual desde la raíz del repo, un notebook o un test.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    path = Path(path)
    cfg_path = path if path.is_absolute() else PROJECT_ROOT / path
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resuelve una ruta del config (posiblemente relativa) contra la raíz del proyecto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
