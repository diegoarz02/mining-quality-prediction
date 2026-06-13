"""Ensembles sobre los modelos del stack: blend ponderado y stacking temporal.

El blend optimiza pesos no negativos (suma 1) minimizando el MAE de validación. El
stacking entrena un meta-learner Ridge sobre predicciones out-of-fold generadas con
walk-forward dentro de train (con purga), de modo que el meta nunca ve predicciones
hechas sobre datos que el modelo base ya conocía. Por política, un ensemble solo
reemplaza al mejor modelo individual si lo supera por más de 2% de MAE relativo en
validación: a igualdad gana el modelo simple por defendibilidad y despliegue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from src.models.cv import walk_forward_folds


def optimize_blend_weights(preds_val: dict[str, np.ndarray], y_val: pd.Series) -> dict[str, float]:
    """Pesos del blend que minimizan el MAE en validación (restricción simplex)."""
    names = list(preds_val)
    matrix = np.column_stack([preds_val[n] for n in names])
    y = np.asarray(y_val, dtype=float)

    def mae(weights: np.ndarray) -> float:
        return float(np.abs(matrix @ weights - y).mean())

    n = len(names)
    result = minimize(
        mae, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    return dict(zip(names, [round(float(w), 4) for w in result.x]))


def blend_predict(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return np.sum([weights[n] * preds[n] for n in weights], axis=0)


def temporal_oof_predictions(
    builders: dict[str, callable],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Predicciones out-of-fold walk-forward (expandible con purga) dentro de train.

    Devuelve un frame con una columna por modelo base, solo en las filas que fueron
    bloque de validación de algún fold, y el target alineado.
    """
    oof = {name: pd.Series(index=X.index, dtype=float) for name in builders}
    for tr_idx, va_idx in walk_forward_folds(len(X), n_splits, "expanding", gap):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        for name, builder in builders.items():
            model = builder(X_tr, y_tr, X_va, y_va)
            oof[name].iloc[va_idx] = model.predict(X_va)
    frame = pd.DataFrame(oof).dropna()
    return frame, y.loc[frame.index]


def fit_stacker(oof: pd.DataFrame, y_oof: pd.Series) -> Ridge:
    """Meta-learner Ridge sobre las predicciones out-of-fold temporales."""
    return Ridge(alpha=1.0).fit(oof, y_oof)
