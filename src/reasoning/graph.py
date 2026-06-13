"""Grafo de razonamiento con LangGraph.

Flujo: validar entrada -> (condicional) -> analizar predicción -> drivers SHAP ->
escenarios what-if -> síntesis de conclusiones. Si la validación de entrada falla, una
arista condicional desvía a un nodo de advertencia en lugar de producir conclusiones.
"""
from __future__ import annotations

import functools

import shap
from langgraph.graph import END, START, StateGraph

from src.config import load_config
from src.models.delta import DeltaTargetRegressor
from src.models.train import fit_winner, lag1_feature
from src.reasoning.llm import make_llm
from src.reasoning.nodes import (
    ReasoningContext,
    analyze_drivers,
    analyze_prediction,
    evaluate_scenarios,
    synthesize,
    validate_input,
    warn,
)
from src.reasoning.state import ReasoningState
from src.simulation.scenarios import reference_state


def build_context(config: dict | None = None) -> ReasoningContext:
    cfg = config or load_config()
    model, data = fit_winner(cfg)
    X = data["X_train"]
    target = cfg["data"]["target"]
    manipulable = [c for c in X.columns if c.endswith("__mean") and any(
        k in c for k in ["Amina Flow", "Starch Flow", "Air Flow", "Level", "Ore Pulp pH"])]
    # Variables que un operador entregaría: feeds de laboratorio, medias horarias de los
    # sensores y la última sílice conocida. Los features derivados (lags, EWM, ratios)
    # se completan internamente desde el punto de referencia.
    feeds = [c for c in cfg["data"]["feed_cols"] if c in X.columns]
    expected = feeds + [c for c in X.columns if c.endswith("__mean")] + [lag1_feature(cfg)]
    is_delta = isinstance(model, DeltaTargetRegressor)
    # Para el modelo delta, SHAP explica el booster interno: las atribuciones quedan en
    # escala de cambio de la sílice respecto a la hora anterior.
    explainer = shap.TreeExplainer(model.base_model if is_delta else model)
    return ReasoningContext(
        model=model,
        feature_names=list(X.columns),
        X_ref=X,
        explainer=explainer,
        manipulable=manipulable,
        target_range=data["target_range"],
        target=target,
        silica_lag_feature=lag1_feature(cfg),
        llm=make_llm(),
        is_delta=is_delta,
        expected_inputs=expected,
        reference=reference_state(X),
    )


def _route(state: dict) -> str:
    return state.get("route", "ok")


def build_graph(ctx: ReasoningContext):
    g = StateGraph(ReasoningState)
    g.add_node("validate", functools.partial(validate_input, ctx=ctx))
    g.add_node("predict", functools.partial(analyze_prediction, ctx=ctx))
    g.add_node("drivers", functools.partial(analyze_drivers, ctx=ctx))
    g.add_node("scenarios", functools.partial(evaluate_scenarios, ctx=ctx))
    g.add_node("synthesize", functools.partial(synthesize, ctx=ctx))
    g.add_node("warn", functools.partial(warn, ctx=ctx))

    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", _route, {"ok": "predict", "fail": "warn"})
    g.add_edge("predict", "drivers")
    g.add_edge("drivers", "scenarios")
    g.add_edge("scenarios", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("warn", END)
    return g.compile()


def default_operating_point(ctx: ReasoningContext) -> dict:
    return reference_state(ctx.X_ref).to_dict()


def run_reasoning(input_features: dict | None = None, config: dict | None = None) -> dict:
    ctx = build_context(config)
    graph = build_graph(ctx)
    if input_features is None:
        input_features = default_operating_point(ctx)
    return graph.invoke({"input_features": input_features})
