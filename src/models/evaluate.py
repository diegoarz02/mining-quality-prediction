"""Métricas de regresión y utilidades de residuos compartidas por notebooks y entrenamiento."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true, y_pred, target_range: float | None = None
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(mask.sum()),
    }
    if target_range:
        # MAE como proporción del rango observado del target, más legible operacionalmente.
        metrics["mae_pct_range"] = 100.0 * metrics["mae"] / target_range
    return metrics


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Arma los diccionarios de métricas por modelo en una tabla comparativa ordenada por MAE."""
    table = pd.DataFrame(results).T
    return table.sort_values("mae")


def residual_frame(index: pd.Index, y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred}, index=index
    )
