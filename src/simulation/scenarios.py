"""Análisis de escenarios what-if sobre el modelo entrenado.

El modelo es asociativo, así que estas son sensibilidades direccionales, no garantías de
control. Los barridos se mantienen dentro del rango histórico observado de cada variable
(nunca extrapolan) y cambian un setpoint de la hora actual a la vez, manteniendo fijo
todo lo demás (incluida la trayectoria reciente de la sílice).

Además del what-if puntual (efecto sobre la próxima hora), `simulate_sustained` itera el
modelo delta de forma autorregresiva para estimar el efecto acumulado de sostener un
setpoint varias horas: la predicción de cada hora alimenta los features de historia del
target de la siguiente. Los errores se acumulan paso a paso, así que la trayectoria es
una aproximación direccional, no una garantía de control.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def reference_state(X: pd.DataFrame, window: int = 720) -> pd.Series:
    """Un punto de operación representativo: la mediana de las últimas `window` horas."""
    return X.iloc[-window:].median()


def predict_point(model, point: pd.Series) -> float:
    return float(model.predict(point.to_frame().T)[0])


def observed_range(X: pd.DataFrame, feature: str, lo: float = 5, hi: float = 95) -> tuple[float, float]:
    return tuple(np.percentile(X[feature].to_numpy(), [lo, hi]))


def sweep_feature(model, reference: pd.Series, X: pd.DataFrame, feature: str, n: int = 21) -> pd.DataFrame:
    lo, hi = observed_range(X, feature)
    base = predict_point(model, reference)
    rows = []
    for v in np.linspace(lo, hi, n):
        point = reference.copy()
        point[feature] = v
        rows.append({"value": v, "pred": predict_point(model, point)})
    out = pd.DataFrame(rows)
    out["delta_vs_ref"] = out["pred"] - base
    return out


def target_history_features(history: pd.Series, feature_names: list[str], target: str) -> dict[str, float]:
    """Recalcula los features derivados del target a partir de una historia de sílice.

    Reconoce los patrones usados en la ingeniería de features (lags, momentum y
    rolling de lags estrictamente pasados) y devuelve solo los que existan en
    ``feature_names``, de modo que funciona igual con el set completo o el podado.
    """
    s = pd.Series(history, dtype=float)
    out: dict[str, float] = {}
    for name in feature_names:
        if not name.startswith(target):
            continue
        if m := re.fullmatch(rf"{re.escape(target)}__lag(\d+)h", name):
            out[name] = float(s.iloc[-int(m.group(1))])
        elif name == f"{target}__mom1h":
            out[name] = float(s.iloc[-1] - s.iloc[-2])
        elif m := re.fullmatch(rf"{re.escape(target)}__lagroll(\d+)h_(mean|std)", name):
            window = s.iloc[-int(m.group(1)):]
            out[name] = float(window.mean() if m.group(2) == "mean" else window.std())
    return out


def simulate_sustained(
    model,
    reference: pd.Series,
    history: pd.Series,
    target: str,
    overrides: dict[str, float] | None = None,
    horizon: int = 8,
) -> pd.Series:
    """Trayectoria de sílice al sostener un punto de operación durante ``horizon`` horas.

    Las variables de proceso quedan fijas en ``reference`` (más los ``overrides`` de la
    palanca simulada); en cada paso, los features de historia del target se recalculan
    desde la trayectoria simulada y la predicción se anexa a la historia. Comparar la
    trayectoria con overrides contra la trayectoria base (sin overrides) aísla el efecto
    acumulado del setpoint bajo el supuesto ceteris paribus.
    """
    point = reference.copy()
    for feature, value in (overrides or {}).items():
        point[feature] = value
    sim_history = history.astype(float).copy()
    preds = []
    for _ in range(horizon):
        for name, value in target_history_features(sim_history, list(point.index), target).items():
            point[name] = value
        pred = predict_point(model, point)
        preds.append(pred)
        sim_history = pd.concat([sim_history, pd.Series([pred])], ignore_index=True)
    return pd.Series(preds, index=pd.RangeIndex(1, horizon + 1, name="hora"))


def scenario_scan(model, reference: pd.Series, X: pd.DataFrame, features: list[str], n: int = 21) -> pd.DataFrame:
    """Para cada variable manipulable, barre su rango observado y reporta la oscilación de la sílice."""
    base = predict_point(model, reference)
    rows = []
    for feature in features:
        if feature not in X.columns:
            continue
        sweep = sweep_feature(model, reference, X, feature, n)
        lo_pred, hi_pred = sweep["pred"].iloc[0], sweep["pred"].iloc[-1]
        rows.append({
            "feature": feature,
            "ref_value": float(reference[feature]),
            "range_low": float(sweep["value"].iloc[0]),
            "range_high": float(sweep["value"].iloc[-1]),
            "silica_at_low": lo_pred,
            "silica_at_high": hi_pred,
            "delta_low_to_high": hi_pred - lo_pred,
            "max_abs_delta_vs_ref": float(sweep["delta_vs_ref"].abs().max()),
        })
    return pd.DataFrame(rows).sort_values("max_abs_delta_vs_ref", ascending=False).reset_index(drop=True)
