"""Esquemas de validación cruzada temporal comparables.

Tres variantes de walk-forward sobre una serie ordenada cronológicamente:

- ``expanding``: ventana de entrenamiento expandible (equivalente a TimeSeriesSplit).
- ``sliding``: ventana de entrenamiento deslizante de tamaño fijo.
- ``*_gap``: cualquiera de las anteriores con purga (embargo) de ``gap`` horas entre el
  final del train y el inicio de la validación del fold. Con lags del target de hasta
  3 h, una purga de 3 h garantiza que ningún feature del primer punto de validación
  haya visto observaciones del tramo de entrenamiento inmediato.

Los folds se definen por posición (la serie es horaria y está ordenada), con bloques de
validación contiguos de igual tamaño que cubren el tramo final de la serie.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.models.evaluate import regression_metrics


def walk_forward_folds(
    n_obs: int,
    n_splits: int = 5,
    scheme: str = "expanding",
    gap: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Genera índices posicionales (train, val) para cada fold.

    ``scheme`` admite ``expanding`` o ``sliding``. El tamaño del bloque de validación
    es ``n_obs // (n_splits + 1)``, igual que TimeSeriesSplit; la ventana deslizante
    fija su tamaño en el del primer train expandible para que ambos esquemas vean los
    mismos bloques de validación y solo cambie la historia disponible.
    """
    val_size = n_obs // (n_splits + 1)
    first_train_end = n_obs - n_splits * val_size
    window = first_train_end - gap
    if window <= 0:
        raise ValueError("No hay suficientes observaciones para el esquema pedido.")
    for k in range(n_splits):
        val_start = first_train_end + k * val_size
        val_end = val_start + val_size
        train_end = val_start - gap
        train_start = 0 if scheme == "expanding" else max(0, train_end - window)
        yield (
            np.arange(train_start, train_end),
            np.arange(val_start, min(val_end, n_obs)),
        )


def cross_validate_temporal(
    builder,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scheme: str = "expanding",
    gap: int = 0,
) -> pd.DataFrame:
    """Evalúa un modelo fold a fold con el esquema pedido.

    ``builder(X_tr, y_tr, X_va, y_va)`` debe devolver un modelo ya ajustado cuyo
    ``predict`` esté en la escala del nivel (los wrappers delta reconstruyen el nivel
    internamente). Devuelve una tabla con MAE/RMSE/R2 por fold.
    """
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(
        walk_forward_folds(len(X), n_splits, scheme, gap)
    ):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = builder(X_tr, y_tr, X_va, y_va)
        metrics = regression_metrics(y_va, model.predict(X_va))
        rows.append({"fold": fold, "n_train": len(tr_idx), "n_val": len(va_idx), **metrics})
    return pd.DataFrame(rows).set_index("fold")


def cv_summary(folds_table: pd.DataFrame) -> dict[str, float]:
    """Media y desviación del MAE entre folds (estabilidad del esquema)."""
    return {
        "mae_mean": float(folds_table["mae"].mean()),
        "mae_std": float(folds_table["mae"].std()),
        "r2_mean": float(folds_table["r2"].mean()),
        "r2_std": float(folds_table["r2"].std()),
    }
