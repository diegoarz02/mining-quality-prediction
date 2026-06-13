"""Explicabilidad SHAP para el modelo de árboles.

TreeExplainer entrega atribuciones exactas y aditivas para LightGBM. Estas utilidades
producen una tabla global de drivers (magnitud y dirección) y localizan casos
contrastantes para explicaciones locales; las reutiliza el grafo de razonamiento (Nivel 5).
Si el modelo es delta, las atribuciones se interpretan en escala de cambio respecto a la
hora anterior (qué empuja la sílice hacia arriba o abajo frente a la última medición).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    return explainer, explanation


def driver_table(explanation, X: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Ordena los features por |SHAP| medio y les añade una dirección operacional.

    direction = signo de corr(valor del feature, valor SHAP): +1 significa que un valor
    mayor del feature empuja la sílice predicha hacia arriba; -1, hacia abajo.
    """
    values = explanation.values
    names = list(explanation.feature_names)
    mean_abs = np.abs(values).mean(axis=0)
    directions = []
    for j in range(values.shape[1]):
        fv = X.iloc[:, j].to_numpy()
        sv = values[:, j]
        if np.std(fv) < 1e-12 or np.std(sv) < 1e-12:
            directions.append(0.0)
        else:
            directions.append(float(np.sign(np.corrcoef(fv, sv)[0, 1])))
    table = pd.DataFrame(
        {"feature": names, "mean_abs_shap": mean_abs, "direction": directions}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return table.head(k) if k else table


def contrasting_cases(model, X: pd.DataFrame) -> dict[str, object]:
    """Índices de las filas con sílice predicha más alta y más baja, para explicaciones locales."""
    preds = model.predict(X)
    return {
        "high": X.index[int(np.argmax(preds))],
        "low": X.index[int(np.argmin(preds))],
        "high_pred": float(np.max(preds)),
        "low_pred": float(np.min(preds)),
    }
