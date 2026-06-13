"""Corrección del sesgo en sílice alta.

El modelo de nivel subestima el tercil alto del target (regresión a la media), que es el
régimen operacionalmente crítico. Aquí se comparan variantes de objetivo y de pesos de
muestra; la regla de decisión es que el candidato ganador no puede empeorar
materialmente el MAE del tercil alto aunque gane en MAE global.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tercile_report(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """MAE y sesgo (real - predicho) por tercil del valor real de sílice."""
    frame = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=float),
        "residual": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
    })
    frame["tercil"] = pd.qcut(frame["y_true"], 3, labels=["bajo", "medio", "alto"])
    return frame.groupby("tercil", observed=True)["residual"].agg(
        mae=lambda s: float(s.abs().mean()),
        sesgo="mean",
        n="size",
    )


def silica_sample_weight(y: pd.Series, strength: float = 1.0) -> np.ndarray:
    """Peso creciente con el nivel de sílice: 1 en el mínimo, 1 + strength en el máximo."""
    y = np.asarray(y, dtype=float)
    span = y.max() - y.min()
    return 1.0 + strength * (y - y.min()) / span


def bias_variants(base_params: dict) -> dict[str, dict]:
    """Variantes de objetivo de LightGBM a comparar contra el default (L2 sobre el delta)."""
    return {
        "objetivo_l2": dict(base_params),
        "objetivo_l1": {**base_params, "objective": "regression_l1"},
        "objetivo_huber": {**base_params, "objective": "huber", "alpha": 0.9},
    }
