"""Nodos del grafo de razonamiento.

Cada nodo actualiza el estado tipado. El LLM, cuando está disponible, solo narra números
que los nodos previos ya calcularon; el nodo de síntesis verifica que toda cifra decimal
del texto corresponda a un valor presente en el estado y marca cualquier cosa no
verificable. La validación de entrada se hace contra las variables de operación esperadas
por el grafo (sensores base, feeds y sílice reciente), no contra los features internos
derivados, que se completan desde el punto de referencia.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.reasoning.llm import call_with_backoff
from src.simulation.scenarios import predict_point, sweep_feature

# Límites físicos por patrón de nombre de variable: una entrada fuera de estos rangos no
# es un punto de operación raro sino un dato imposible (sensor dañado o error de carga).
PHYSICAL_BOUNDS: list[tuple[str, float, float]] = [
    ("Ore Pulp pH", 0.0, 14.0),
    ("% Iron Feed", 0.0, 100.0),
    ("% Silica Feed", 0.0, 100.0),
    ("% Silica Concentrate", 0.0, 100.0),
    ("Flow", 0.0, float("inf")),
    ("Level", 0.0, float("inf")),
    ("Density", 0.0, 10.0),
]


@dataclass
class ReasoningContext:
    model: object
    feature_names: list[str]
    X_ref: pd.DataFrame
    explainer: object
    manipulable: list[str]
    target_range: float
    target: str
    silica_lag_feature: str
    llm: object | None = None
    is_delta: bool = False
    # Variables de operación que el grafo espera recibir; el resto de features
    # derivados se completa internamente desde la mediana de referencia.
    expected_inputs: list[str] = field(default_factory=list)
    reference: pd.Series | None = None


def _row(features: dict, ctx: ReasoningContext) -> pd.Series:
    """Arma el vector completo de features: lo recibido más la referencia para el resto."""
    ref = ctx.reference if ctx.reference is not None else ctx.X_ref.median()
    return pd.Series(
        {c: features.get(c, float(ref[c])) for c in ctx.feature_names}, dtype=float
    )


def _physical_bounds(name: str) -> tuple[float, float] | None:
    for pattern, lo, hi in PHYSICAL_BOUNDS:
        if pattern in name:
            return lo, hi
    return None


def validate_input(state: dict, ctx: ReasoningContext) -> dict:
    """Valida el punto de operación contra las variables esperadas y sus rangos.

    Tres chequeos, cada uno con mensaje específico: variables esperadas ausentes,
    valores físicamente imposibles y valores fuera del rango histórico observado.
    """
    feats = state.get("input_features", {})
    issues: list[str] = []

    expected = ctx.expected_inputs or ctx.feature_names
    missing = [c for c in expected if c not in feats or pd.isna(feats.get(c))]
    if missing:
        listed = ", ".join(missing[:5])
        issues.append(f"faltan variables de operación requeridas: {listed}")

    # Los chequeos de rango aplican solo a las variables de operación que entrega el
    # ingeniero: las derivadas (diferencias, pendientes, ratios) pueden ser negativas
    # por construcción y no son entradas directas.
    lo_hist, hi_hist = ctx.X_ref.min(), ctx.X_ref.max()
    for name, value in feats.items():
        if name not in expected or name not in ctx.feature_names or pd.isna(value):
            continue
        bounds = _physical_bounds(name)
        if bounds and not (bounds[0] <= value <= bounds[1]):
            issues.append(
                f"{name} = {value:g} es físicamente imposible "
                f"(rango válido: {bounds[0]:g} a {bounds[1]:g})"
            )
        elif value < lo_hist[name] or value > hi_hist[name]:
            issues.append(
                f"{name} = {value:g} está fuera del rango histórico observado "
                f"({lo_hist[name]:.2f} a {hi_hist[name]:.2f})"
            )

    ok = not issues
    return {"quality_ok": ok, "quality_issues": issues, "route": "ok" if ok else "fail"}


def analyze_prediction(state: dict, ctx: ReasoningContext) -> dict:
    row = _row(state["input_features"], ctx)
    return {
        "prediction": predict_point(ctx.model, row),
        "baseline_persistence": float(row[ctx.silica_lag_feature]),
    }


def analyze_drivers(state: dict, ctx: ReasoningContext) -> dict:
    """Atribuciones SHAP locales. Para el modelo delta están en escala de cambio:
    cuánto empuja cada variable a la sílice respecto a la última hora conocida."""
    row = _row(state["input_features"], ctx)
    contrib = ctx.explainer(row.to_frame().T).values[0]
    order = np.argsort(np.abs(contrib))[::-1][:5]
    drivers = [
        {"feature": ctx.feature_names[i], "shap": round(float(contrib[i]), 4),
         "direction": "sube" if contrib[i] > 0 else "baja"}
        for i in order
    ]
    return {"drivers": drivers}


def evaluate_scenarios(state: dict, ctx: ReasoningContext) -> dict:
    row = _row(state["input_features"], ctx)
    base = predict_point(ctx.model, row)
    best = row.copy()
    ranked = []
    for feature in ctx.manipulable:
        sweep = sweep_feature(ctx.model, row, ctx.X_ref, feature, n=15)
        best[feature] = sweep.loc[sweep["pred"].idxmin(), "value"]
        ranked.append({"feature": feature, "max_abs_delta": round(float(sweep["delta_vs_ref"].abs().max()), 4)})
    best_pred = predict_point(ctx.model, best)
    ranked.sort(key=lambda d: d["max_abs_delta"], reverse=True)
    return {"scenarios": ranked[:5], "best_case_silica": round(best_pred, 4),
            "achievable_reduction": round(base - best_pred, 4)}


def _impact(state: dict, ctx: ReasoningContext) -> dict:
    pred, base = state["prediction"], state["baseline_persistence"]
    reduction = state.get("achievable_reduction", 0.0)
    return {
        "predicted_silica": round(pred, 3),
        "persistence_baseline": round(base, 3),
        "vs_baseline": round(pred - base, 3),
        "achievable_reduction": round(reduction, 3),
        "achievable_reduction_pct_range": round(100 * reduction / ctx.target_range, 2),
    }


def _deterministic_conclusions(state: dict, impact: dict) -> str:
    """Síntesis por reglas, con el mismo formato de informe que la versión LLM."""
    drivers = state["drivers"]
    top_lever = state["scenarios"][0]["feature"] if state.get("scenarios") else "ninguna"
    return (
        f"Situación: entrada validada; la sílice predicha para la próxima hora es "
        f"{impact['predicted_silica']}%, frente a {impact['persistence_baseline']}% de la última "
        f"medición de laboratorio (cambio esperado de {impact['vs_baseline']} puntos). "
        f"Drivers principales: {', '.join(d['feature'] for d in drivers[:3])}. "
        f"Escenarios: la palanca manipulable más sensible es {top_lever}; ajustando los setpoints "
        f"dentro del rango histórico, el modelo estima reducir la sílice hasta "
        f"{impact['achievable_reduction']} puntos ({impact['achievable_reduction_pct_range']}% del rango). "
        f"Riesgos y límites: la estimación es asociativa, no causal, y no sustituye la evaluación "
        f"del ingeniero de turno."
    )


def _llm_conclusions(state: dict, ctx: ReasoningContext, impact: dict) -> str:
    """Párrafo de situación grounded. Los drivers, escenarios y la recomendación se
    estructuran aparte (de forma determinística), así que aquí el LLM solo redacta el
    resumen situacional; las cifras provienen del estado del grafo."""
    facts = (
        f"prediccion_silica={impact['predicted_silica']}; "
        f"ultima_medicion_laboratorio={impact['persistence_baseline']}; "
        f"cambio_esperado_vs_ultima_medicion={impact['vs_baseline']}; "
        f"reduccion_alcanzable={impact['achievable_reduction']} "
        f"({impact['achievable_reduction_pct_range']}% del rango del target)"
    )
    system = (
        "Eres analista de procesos de una planta de flotación de mineral de hierro y reportas al equipo "
        "de operaciones. El sistema predice el porcentaje de sílice (impureza) del concentrado una hora "
        "antes de que el laboratorio lo confirme, para que los ingenieros actúen temprano: menos mineral "
        "fuera de especificación termina en relaves y la calidad se corrige antes. "
        "Reglas de fundamentación, obligatorias: cita únicamente cifras presentes en los datos que se te "
        "entregan, sin inventar ni redondear distinto; declara los supuestos (retraso de laboratorio de "
        "una hora, modelo asociativo); distingue asociación de causalidad. "
        "Redacta un único párrafo de situación (la predicción frente a la última medición y su lectura "
        "operacional). No enumeres drivers ni escenarios: se muestran por separado. "
        "Estilo: español sobrio y directo, con tildes correctas, sin emojis, sin signos de exclamación y "
        "sin guiones largos. Máximo 70 palabras."
    )
    human = f"Redacta el párrafo de situación a partir de estos resultados del modelo:\n{facts}"
    return call_with_backoff(ctx.llm, [("system", system), ("human", human)]).content.strip()


def _structured_report(state: dict, impact: dict, situation: str) -> dict:
    """Informe estructurado para la interfaz: situación (narrada) más drivers, escenarios,
    recomendación y riesgos, todos derivados de las cifras del estado del grafo."""
    drivers = [
        f"{d['feature']}: {d['direction']} la sílice (SHAP {d['shap']:+.4f})"
        for d in state.get("drivers", [])[:5]
    ]
    scenarios = [
        f"{s['feature']}: efecto máximo estimado {s['max_abs_delta']} puntos en el rango observado"
        for s in state.get("scenarios", [])[:3]
    ]
    top = state["scenarios"][0]["feature"] if state.get("scenarios") else None
    action = (
        f"Ajustar {top} dentro de su rango histórico observado y vigilar la respuesta de la sílice."
        if top else
        "Ninguna palanca mostró efecto material en el rango observado; mantener el punto de operación."
    )
    recommendation = {
        "impact": (
            f"reducción estimada de hasta {impact['achievable_reduction']} puntos de sílice "
            f"({impact['achievable_reduction_pct_range']}% del rango)"
        ),
        "action": action,
    }
    risks = [
        "El modelo es asociativo (correlacional): las sensibilidades no garantizan control causal.",
        "Válido solo dentro del rango histórico observado; no se extrapola.",
        "No sustituye la evaluación del ingeniero de turno ni considera costos de reactivos.",
    ]
    return {
        "situation": situation,
        "drivers": drivers,
        "scenarios": scenarios,
        "recommendation": recommendation,
        "risks_and_limits": risks,
    }


def _verify_figures(text: str, state: dict, impact: dict) -> list[float]:
    """Toda cifra decimal del texto debe existir en el estado del grafo."""
    allowed = {round(float(v), 3) for v in impact.values()}
    allowed |= {round(float(d["shap"]), 4) for d in state.get("drivers", [])}
    allowed |= {round(float(s["max_abs_delta"]), 4) for s in state.get("scenarios", [])}
    allowed.add(round(float(state.get("best_case_silica", 0.0)), 4))
    unverified = []
    for token in re.findall(r"-?\d+\.\d+", text):  # solo decimales: mediciones afirmadas
        value = float(token)
        if not any(abs(value - a) <= 0.06 for a in allowed):
            unverified.append(value)
    return unverified


def synthesize(state: dict, ctx: ReasoningContext) -> dict:
    impact = _impact(state, ctx)
    warnings: list[str] = []
    used_llm = False
    if ctx.llm is not None:
        try:
            text = _llm_conclusions(state, ctx, impact)
            used_llm = True
        except Exception as exc:  # noqa: BLE001
            text = _deterministic_conclusions(state, impact)
            warnings.append(f"LLM no disponible ({type(exc).__name__}); se usó la síntesis determinística")
    else:
        text = _deterministic_conclusions(state, impact)
    unverified = _verify_figures(text, state, impact)
    if unverified:
        warnings.append(f"cifras no verificables en el texto: {unverified}")
    report = _structured_report(state, impact, text)
    return {
        "impact_estimate": impact,
        "conclusions": text,
        "report": report,
        "used_llm": used_llm,
        "is_fallback": not used_llm,
        "warnings": warnings,
    }


def warn(state: dict, ctx: ReasoningContext) -> dict:
    """Nodo de advertencia: explica qué variables fallaron la validación y por qué."""
    issues = state.get("quality_issues", [])
    detail = " ".join(f"({i + 1}) {issue}." for i, issue in enumerate(issues))
    text = (
        "No se generaron conclusiones operacionales porque la validación de la entrada falló. "
        f"Problemas detectados: {detail} "
        "Corrija estos valores y vuelva a ejecutar el análisis."
    )
    return {
        "conclusions": text,
        "report": {"error": "Entrada inválida", "situation": text},
        "impact_estimate": {},
        "used_llm": False,
        "is_fallback": True,
        "warnings": ["entrada inválida: el grafo se desvió al nodo de advertencia"],
    }
