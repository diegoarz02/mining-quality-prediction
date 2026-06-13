"""Estado tipado que se pasa entre los nodos del grafo de razonamiento."""
from __future__ import annotations

from typing import TypedDict


class ReasoningState(TypedDict, total=False):
    # Punto de operación de entrada y su veredicto de calidad de datos
    input_features: dict
    quality_ok: bool
    quality_issues: list[str]
    route: str
    # Predicción del modelo y referencia ingenua
    prediction: float
    baseline_persistence: float
    # Drivers SHAP locales de esta predicción
    drivers: list[dict]
    # Resultados what-if alrededor de este punto de operación
    scenarios: list[dict]
    best_case_silica: float
    achievable_reduction: float
    # Síntesis final
    impact_estimate: dict
    conclusions: str
    report: dict
    warnings: list[str]
    used_llm: bool
    is_fallback: bool
