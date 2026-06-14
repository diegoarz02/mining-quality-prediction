"""Servicio FastAPI de inferencia de % de sílice en concentrado.

El modelo se carga en 3 niveles de fallback:
  1. Registry MLflow (alias ``champion``) - modo normal en local y Render con BD.
  2. Artefacto empaquetado en ``models/champion/`` (booster nativo + metadata.json +
     sample_hourly.parquet) - modo standalone para Render sin mlflow.db/mlruns/.
  3. ``fit_winner(cfg)`` - reentrenamiento completo como último recurso.

Los clientes pueden enviar un punto de operación parcial (las features faltantes se
completán con la referencia) o la historia horaria cruda, con la que el servicio
construye las features él mismo.

Lanzar siempre desde el venv del proyecto:  python -m uvicorn api.main:app
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envcheck import ensure_dependencies

ensure_dependencies(["fastapi", "pandas", "lightgbm", "xgboost", "mlflow", "sklearn", "yaml"])

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    FeaturesResponse,
    HealthResponse,
    PredictFromHistoryRequest,
    PredictFromHistoryResponse,
    PredictRequest,
    PredictResponse,
    SimulateRequest,
    SimulateResponse,
    ReferenceResponse,
    ExplainResponse,
    ExplainContribution,
    ReportRequest,
    ReportResponse,
    SimulateSustainedResponse,
    SimulateSustainedRequest,
)
from src.config import load_config
from src.features.build import build_inference_features, load_selected_features
from src.models.train import fit_winner, lag1_feature, prepare_data
from src.simulation.scenarios import observed_range, simulate_sustained
import shap

STATE: dict = {}

DESCRIPTION = """\
Servicio de predicción horaria del **% de sílice (impureza) en el concentrado** de una
planta de flotación de mineral de hierro.

En la planta, la calidad del concentrado se confirma por laboratorio con retraso. Este
servicio anticipa ese resultado con las variables de proceso disponibles en tiempo real
y la última sílice conocida (retraso de laboratorio asumido de ~1 hora), para que los
ingenieros de operaciones actúen antes: menos mineral fuera de especificación termina en
relaves.

Endpoints:

- `POST /predict-from-history`: el modo recomendado. Recibe las últimas horas crudas de
  la planta y construye las features internamente.
- `POST /predict`: modo simplificado. Acepta un punto de operación parcial; lo no
  enviado se completa con la referencia.
- `POST /simulate`: escenario what-if sobre un punto de operación (sensibilidades
  asociativas, no causales).
- `GET /health`, `GET /features`: estado del servicio y catálogo de features.
"""

TAGS = [
    {"name": "Estado del servicio", "description": "Disponibilidad del modelo y catálogo de features."},
    {"name": "Predicción", "description": "Predicción de la sílice del concentrado para la hora en curso."},
    {"name": "Simulación", "description": "Escenarios what-if alrededor de un punto de operación."},
]


def _load_state() -> None:
    cfg = load_config()
    win = cfg.get("winner", {})
    source = "uninitialized"
    
    try:
        from src.models.registry import load_champion
        from src.features.build import load_selected_features
        from src.models.train import prepare_data
        
        subset = load_selected_features(cfg) if win.get("use_selected_features") else None
        data = prepare_data(cfg, feature_subset=subset)
        X = data["X_train"]
        hourly_data = data["hourly"]
        
        model = load_champion(cfg)
        source = "loaded_champion"
        
        y_pred = model.predict(data["X_test"])
        residuals_test = (data["y_test"] - y_pred).tolist()
        mae_test = float(abs(data["y_test"] - y_pred).mean())
        
    except Exception:
        # Level 2: Artefacto empaquetado
        import json
        from pathlib import Path
        import lightgbm as lgb
        import xgboost as xgb
        from src.models.registry import _DeltaPredictor
        from src.features.build import build_features

        champ_dir = Path("models/champion")
        if champ_dir.exists() and (champ_dir / "metadata.json").exists():
            with open(champ_dir / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            family = meta.get("family", "lightgbm")
            if family == "lightgbm":
                booster = lgb.Booster(model_file=str(champ_dir / "model.txt"))
            elif family == "xgboost":
                booster = xgb.Booster()
                booster.load_model(str(champ_dir / "model.json"))
                
            class _BoosterPyFunc:
                def __init__(self, b): self.b = b
                def predict(self, X):
                    if family == "xgboost":
                        return self.b.predict(xgb.DMatrix(X))
                    return self.b.predict(X)
                @property
                def booster(self):
                    return self.b

            if meta.get("representation") == "delta":
                model = _DeltaPredictor(_BoosterPyFunc(booster), lag1_feature(cfg))
            else:
                model = _BoosterPyFunc(booster)
                
            # Rebuild X and others from sample_hourly.parquet
            hourly_data = pd.read_parquet(champ_dir / "sample_hourly.parquet")
            frame = build_features(hourly_data, cfg)

            # Nombres reales de las features (con espacios), guardados en metadata.json por
            # package_model.py. No se usan los del booster: LightGBM los sanitiza al guardar
            # model.txt y no coincidirían con las columnas de build_features.
            features = meta.get("features")
            if not features:
                from src.features.build import load_selected_features
                features = load_selected_features(cfg)
            X = frame[features].dropna()
            
            mae_test = float(meta.get("test_mae", 0.46))
            residuals_test = meta.get("residuals_test", [])
            source = "loaded_packaged"
        else:
            # Level 3: fit_winner
            from src.models.train import fit_winner, prepare_data, load_selected_features
            
            subset = load_selected_features(cfg) if win.get("use_selected_features") else None
            model, data = fit_winner(cfg)
            X = data["X_train"]
            hourly_data = data["hourly"]
            
            y_pred = model.predict(data["X_test"])
            residuals_test = (data["y_test"] - y_pred).tolist()
            mae_test = float(abs(data["y_test"] - y_pred).mean())
            
            source = "trained_fallback"

    p5 = {}
    p95 = {}
    for col in X.columns:
        lo, hi = observed_range(X, col, 5, 95)
        p5[col] = lo
        p95[col] = hi
        
    is_delta = hasattr(model, "base_model") or hasattr(model, "_model")
    # El booster (de model.txt) sanitiza los nombres de feature al guardarse, pero conserva
    # el orden de entrenamiento, que coincide con X.columns. SHAP atribuye por posición, así
    # que usamos los nombres reales de X en ese mismo orden.
    explainer, n_expected = _build_explainer(model, cfg)
    shap_features = list(X.columns)
    if explainer is not None and n_expected is not None and n_expected != len(shap_features):
        explainer = None

    target = cfg["data"]["target"]
    target_range = float(hourly_data[target].max() - hourly_data[target].min())

    STATE.update({
        "cfg": cfg,
        "model": model,
        "X": X,
        "feature_names": list(X.columns),
        "reference": X.iloc[-720:].median(),
        "min": X.min(),
        "max": X.max(),
        "p5": p5,
        "p95": p95,
        "mae_test": mae_test,
        "residuals_test": residuals_test,
        "source": source,
        "sensors": cfg["data"]["sensor_cols"],
        "feeds": cfg["data"]["feed_cols"],
        "target": target,
        "target_range": target_range,
        "silica_lag_feature": lag1_feature(cfg),
        "hourly_columns": list(hourly_data.columns),
        "explainer": explainer,
        "shap_features": shap_features,
        "is_delta": is_delta,
        "hourly_data": hourly_data,
        "reasoning_graph": None,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield
    STATE.clear()


app = FastAPI(
    title="Predicción de % Sílice en Concentrado - Planta de Flotación",
    description=DESCRIPTION,
    version="2.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Errores de validación de Pydantic traducidos a mensajes accionables en español."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", []) if p != "body")
        received = err.get("input", None)
        problems.append({
            "campo": field or "cuerpo de la solicitud",
            "problema": err.get("msg", "valor inválido"),
            "valor_recibido": received if isinstance(received, (int, float, str, bool, type(None))) else str(type(received).__name__),
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "La solicitud no es válida. Revise los campos indicados y consulte los ejemplos en /docs.",
            "errores": problems,
        },
    )


def _ensure_known(features: dict) -> None:
    unknown = [k for k in features if k not in STATE["feature_names"]]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Features desconocidas: {unknown[:5]}. El modelo espera nombres como "
                f"'{STATE['feature_names'][0]}'; el catálogo completo está en GET /features."
            ),
        )


def _row(features: dict) -> pd.DataFrame:
    reference = STATE["reference"]
    values = {c: features.get(c, float(reference[c])) for c in STATE["feature_names"]}
    return pd.DataFrame([values], columns=STATE["feature_names"])


def _predict(df: pd.DataFrame) -> float:
    return float(STATE["model"].predict(df)[0])


_AGG_LABEL = {"mean": "media", "std": "desv.", "min": "mín", "max": "máx"}

# Features de ingeniería que no siguen el patrón base__agg__transform.
_SPECIAL_LABEL = {
    "hour_sin": "Hora del día (seno)",
    "hour_cos": "Hora del día (coseno)",
    "cross_air_flow_min": "Aire de columnas (mínimo entre columnas)",
    "cross_level_std": "Nivel de columnas (dispersión entre columnas)",
    "ratio_amina_pulp": "Ratio amina/pulpa",
}


def _transform_label(token: str, has_mean: bool) -> str:
    """Frase legible para un sufijo de transformación temporal (sin paréntesis)."""
    m = re.fullmatch(r"lag(\d+)h", token)
    if m:
        return f"{m.group(1)}h atrás"
    m = re.fullmatch(r"(?:lag)?roll(\d+)h_(mean|std)", token)
    if m:
        kind = "media" if m.group(2) == "mean" else "desv."
        return f"{kind} móvil {m.group(1)}h"
    m = re.fullmatch(r"ewm(\d+)h", token)
    if m:
        return f"media, EWM {m.group(1)}h" if has_mean else f"EWM {m.group(1)}h"
    m = re.fullmatch(r"diff(\d+)h", token)
    if m:
        return f"cambio {m.group(1)}h"
    m = re.fullmatch(r"slope(\d+)h", token)
    if m:
        return f"pendiente {m.group(1)}h"
    m = re.fullmatch(r"mom(\d+)h", token)
    if m:
        return f"momentum {m.group(1)}h"
    return token.replace("_", " ")


def _pretty(col: str) -> str:
    """Etiqueta de planta legible para una columna de feature.

    Parsea el nombre por sus separadores ``__`` y arma una única anotación entre
    paréntesis (p. ej. "media móvil 3h"), de modo que nunca queda un paréntesis sin
    cerrar ni un sufijo de ventana colgando.
    """
    if col in _SPECIAL_LABEL:
        return _SPECIAL_LABEL[col]

    parts = col.split("__")
    base, mods = parts[0], parts[1:]
    if not mods:
        return base

    agg = _AGG_LABEL.get(mods[0])
    transforms = mods[1:] if agg else mods
    if not transforms:
        return f"{base} ({agg})" if agg else base

    has_mean = mods[0] == "mean"
    note = ", ".join(_transform_label(t, has_mean) for t in transforms)
    return f"{base} ({note})"


def _build_explainer(model, cfg: dict):
    """TreeExplainer sobre el booster nativo, válido tanto para el modelo del registry
    (envuelto en un pyfunc) como para el empaquetado. Prefiere el artefacto
    ``models/champion/model.txt`` por ser un árbol nativo que SHAP explica directamente.
    Devuelve (explainer, orden_de_features) o (None, None) si el modelo no es de árbol.
    """
    import lightgbm as lgb

    champ = Path(cfg["paths"]["models_dir"]) / "champion" / "model.txt"
    try:
        if champ.exists():
            booster = lgb.Booster(model_file=str(champ))
            return shap.TreeExplainer(booster), booster.num_feature()
    except Exception:
        pass

    base = model
    for attr in ("base_model", "_model"):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break
    if hasattr(base, "unwrap_python_model"):
        try:
            base = base.unwrap_python_model()
        except Exception:
            pass
    if hasattr(base, "booster"):
        base = base.booster
    elif hasattr(base, "booster_"):
        base = base.booster_
    elif hasattr(base, "get_booster"):
        base = base.get_booster()
    try:
        n = base.num_feature() if hasattr(base, "num_feature") else (
            base.num_features() if hasattr(base, "num_features") else None)
        return shap.TreeExplainer(base), n
    except Exception:
        return None, None


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """La raíz redirige a la aplicación web."""
    return RedirectResponse(url="/app/")

web_dir = Path(__file__).resolve().parents[1] / "web"
if not web_dir.exists():
    web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory=str(web_dir), html=True), name="web")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Estado del servicio"],
    summary="Estado del servicio",
    description="Indica si el modelo está cargado, desde dónde se cargó y cuántas features espera.",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_source=STATE.get("source", "uninitialized"),
        n_features=len(STATE.get("feature_names", [])),
    )


@app.get(
    "/features",
    response_model=FeaturesResponse,
    tags=["Estado del servicio"],
    summary="Catálogo de features del modelo",
    description="Nombres exactos de las features que aceptan /predict y /simulate.",
)
def features() -> FeaturesResponse:
    return FeaturesResponse(
        n_features=len(STATE["feature_names"]), features=STATE["feature_names"]
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Predicción"],
    summary="Predicción con punto de operación parcial (modo simplificado)",
    description=(
        "Acepta cualquier subconjunto de features; lo no enviado se completa con el "
        "punto de operación de referencia (mediana de los últimos 30 días de "
        "entrenamiento). Útil para pruebas rápidas; para usar historia real de la "
        "planta, preferir /predict-from-history."
    ),
)
def predict(request: PredictRequest) -> PredictResponse:
    _ensure_known(request.features)
    prediction = _predict(_row(request.features))
    return PredictResponse(
        predicted_silica=round(prediction, 4),
        n_features_provided=len(request.features),
        n_features_filled=len(STATE["feature_names"]) - len(request.features),
    )


def _history_frame(request: PredictFromHistoryRequest) -> pd.DataFrame:
    """Convierte la historia recibida en un frame horario con el esquema del dataset."""
    sensors, feeds, target = STATE["sensors"], STATE["feeds"], STATE["target"]
    known_vars = set(feeds) | set(sensors)
    reference = STATE["reference"]

    rows = []
    index = []
    for i, record in enumerate(request.history):
        unknown = [v for v in record.values if v not in known_vars]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Variables desconocidas en la hora {record.date.isoformat()}: {unknown[:5]}. "
                    f"Se esperan los nombres crudos del proceso: {sorted(known_vars)[:4]}..."
                ),
            )
        is_last = i == len(request.history) - 1
        if not is_last and record.silica is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Falta el resultado de laboratorio (campo 'silica') de la hora "
                    f"{record.date.isoformat()}. Solo la última hora, que es la que se "
                    f"predice, puede omitirlo."
                ),
            )
        row = {}
        for f in feeds:
            row[f] = record.values.get(f, float(reference[f]))
        for s in sensors:
            mean_col = f"{s}__mean"
            row[mean_col] = record.values.get(s, float(reference[mean_col]))
            # La dispersión intra-hora (std/min/max) no puede derivarse de la media; se
            # completa con la referencia y se informa en n_features_filled.
            for stat in ("std", "min", "max"):
                col = f"{s}__{stat}"
                row[col] = float(reference[col]) if col in reference.index else 0.0
        row[target] = record.silica if record.silica is not None else float("nan")
        rows.append(row)
        index.append(record.date.replace(minute=0, second=0, microsecond=0))

    if sorted(index) != index or len(set(index)) != len(index):
        raise HTTPException(
            status_code=422,
            detail="Las horas de 'history' deben venir ordenadas cronológicamente y sin repetidos.",
        )
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="date"))
    return frame


@app.post(
    "/predict-from-history",
    response_model=PredictFromHistoryResponse,
    tags=["Predicción"],
    summary="Predicción desde la historia horaria cruda (modo recomendado)",
    description=(
        "Recibe las últimas N horas de las 21 variables base de la planta más los "
        "resultados de laboratorio ya conocidos, y construye internamente todas las "
        "features temporales (lags, medias móviles, suavizados, ratios). La última hora "
        "de la lista es la que se predice: tiene sensores pero aún no tiene laboratorio. "
        "Con 24 horas de historia todas las features se calculan desde datos reales."
    ),
)
def predict_from_history(request: PredictFromHistoryRequest) -> PredictFromHistoryResponse:
    hourly = _history_frame(request)
    features_frame = build_inference_features(hourly, STATE["cfg"])
    last = features_frame.iloc[-1].reindex(STATE["feature_names"])
    n_missing = int(last.isna().sum())
    last = last.fillna(STATE["reference"])
    prediction = _predict(last.to_frame().T)
    return PredictFromHistoryResponse(
        predicted_silica=round(prediction, 4),
        target_hour=features_frame.index[-1].to_pydatetime(),
        n_hours_received=len(request.history),
        n_features_computed=len(STATE["feature_names"]) - n_missing,
        n_features_filled=n_missing,
    )


@app.post(
    "/simulate",
    response_model=SimulateResponse,
    tags=["Simulación"],
    summary="Escenario what-if sobre un punto de operación",
    description=(
        "Aplica cambios aditivos a variables manipulables y reporta el efecto estimado "
        "sobre la sílice. Las sensibilidades son asociativas (aprendidas de datos "
        "históricos), no garantías causales de control; si una variable queda fuera del "
        "rango histórico, se informa en 'out_of_observed_range'."
    ),
)
def simulate(request: SimulateRequest) -> SimulateResponse:
    _ensure_known(request.base_features)
    _ensure_known(request.deltas)
    base_df = _row(request.base_features)
    base_pred = _predict(base_df)

    sim_df = base_df.copy()
    applied: dict[str, float] = {}
    out_of_range: list[str] = []
    lo, hi = STATE["min"], STATE["max"]
    for feature, delta in request.deltas.items():
        new_value = float(sim_df.at[0, feature] + delta)
        sim_df.at[0, feature] = new_value
        applied[feature] = round(new_value, 4)
        if new_value < lo[feature] or new_value > hi[feature]:
            out_of_range.append(feature)
    sim_pred = _predict(sim_df)
    return SimulateResponse(
        base_silica=round(base_pred, 4),
        simulated_silica=round(sim_pred, 4),
        delta_silica=round(sim_pred - base_pred, 4),
        applied=applied,
        out_of_observed_range=out_of_range,
    )

@app.get("/reference", response_model=ReferenceResponse, tags=["Predicción"])
def reference() -> ReferenceResponse:
    return ReferenceResponse(
        reference=STATE["reference"].to_dict(),
        min=STATE["min"].to_dict(),
        max=STATE["max"].to_dict(),
        p5=STATE["p5"],
        p95=STATE["p95"],
        mae_test=STATE.get("mae_test", 0.46),
        residuals_test=STATE.get("residuals_test", [])
    )

@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["Predicción"],
    summary="Explicación SHAP local de una predicción",
    description=(
        "Devuelve las contribuciones SHAP de la predicción para el punto de operación dado "
        "(features faltantes completadas con la referencia). Para el modelo delta, las "
        "atribuciones explican el cambio respecto a la última sílice conocida; el valor base "
        "reconstruye el nivel sumando ese lag."
    ),
)
def explain(request: PredictRequest) -> ExplainResponse:
    _ensure_known(request.features)
    if STATE.get("explainer") is None:
        raise HTTPException(
            status_code=503,
            detail="El explainer SHAP no está disponible para el modelo cargado.",
        )

    shap_cols = STATE["shap_features"]
    df = _row(request.features)
    prediction = _predict(df)
    df_shap = df[shap_cols]

    shap_values = STATE["explainer"].shap_values(df_shap)
    shap_row = np.asarray(shap_values)[0]
    expected = float(np.ravel(STATE["explainer"].expected_value)[0])
    # Para el modelo delta, SHAP explica el cambio; el nivel base reconstruye sumando la
    # última sílice conocida (el lag de 1 h), de modo que base + contribuciones = predicción.
    if STATE.get("is_delta"):
        expected += float(df_shap.iloc[0][STATE["silica_lag_feature"]])

    contributions = [
        ExplainContribution(
            feature=col,
            label=_pretty(col),
            value=round(float(df_shap.iloc[0, i]), 4),
            shap_value=round(float(shap_row[i]), 4),
        )
        for i, col in enumerate(shap_cols)
    ]
    contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

    return ExplainResponse(
        predicted_silica=round(prediction, 4),
        base_value=round(expected, 4),
        contributions=contributions[:8],
    )

@app.post("/simulate-sustained", response_model=SimulateSustainedResponse, tags=["Simulación"])
def simulate_sustained_endpoint(request: SimulateSustainedRequest) -> SimulateSustainedResponse:
    _ensure_known(request.base_features)
    _ensure_known(request.deltas)
    
    base_df = _row(request.base_features)
    ref_series = base_df.iloc[0]
    
    history_series = STATE["hourly_data"][STATE["target"]]
    
    # Base simulation
    traj_base = simulate_sustained(
        model=STATE["model"],
        reference=ref_series,
        history=history_series,
        target=STATE["target"],
        overrides=None,
        horizon=8
    )
    
    # Delta simulation
    applied = {}
    for k, v in request.deltas.items():
        applied[k] = ref_series[k] + v
        
    traj_sim = simulate_sustained(
        model=STATE["model"],
        reference=ref_series,
        history=history_series,
        target=STATE["target"],
        overrides=applied,
        horizon=8
    )
    
    return SimulateSustainedResponse(
        trajectory=[round(x, 4) for x in traj_sim.tolist()],
        trajectory_base=[round(x, 4) for x in traj_base.tolist()],
        delta_accumulated=round(sum(traj_sim - traj_base), 4)
    )

def _get_reasoning_graph():
    """Compila el grafo de razonamiento una sola vez, reutilizando el modelo y el explainer
    ya cargados en STATE (no reentrena ni vuelve a leer datos: clave para Render standalone)."""
    if STATE.get("reasoning_graph") is None:
        from src.reasoning.graph import build_graph
        from src.reasoning.llm import make_llm
        from src.reasoning.nodes import ReasoningContext

        cfg = STATE["cfg"]
        feature_names = STATE["feature_names"]
        manipulable = [c for c in feature_names if c.endswith("__mean") and any(
            k in c for k in ["Amina Flow", "Starch Flow", "Air Flow", "Level", "Ore Pulp pH"])]
        feeds = [c for c in cfg["data"]["feed_cols"] if c in feature_names]
        expected_inputs = feeds + [c for c in feature_names if c.endswith("__mean")] + [STATE["silica_lag_feature"]]
        ctx = ReasoningContext(
            model=STATE["model"],
            feature_names=feature_names,
            X_ref=STATE["X"],
            explainer=STATE["explainer"],
            manipulable=manipulable,
            target_range=STATE["target_range"],
            target=STATE["target"],
            silica_lag_feature=STATE["silica_lag_feature"],
            llm=make_llm(),
            is_delta=STATE["is_delta"],
            expected_inputs=expected_inputs,
            reference=STATE["reference"],
        )
        STATE["reasoning_graph"] = build_graph(ctx)
    return STATE["reasoning_graph"]


@app.post(
    "/report",
    response_model=ReportResponse,
    tags=["Simulación"],
    summary="Informe operacional del agente de razonamiento (LangGraph)",
    description=(
        "Ejecuta el grafo de razonamiento sobre el punto de operación (validación, "
        "predicción, drivers SHAP, escenarios y síntesis) y devuelve un informe "
        "estructurado. Si el LLM no está disponible o se excede el límite de tasa, se usa "
        "la síntesis determinística por reglas y se marca en 'is_fallback'."
    ),
)
def report(request: ReportRequest) -> ReportResponse:
    full_features = _row(request.features).iloc[0].to_dict()
    try:
        result = _get_reasoning_graph().invoke({"input_features": full_features})
        return ReportResponse(
            report=result.get("report", {}),
            is_fallback=bool(result.get("is_fallback", False)),
            warnings=list(result.get("warnings", []) or []),
        )
    except Exception as exc:  # noqa: BLE001 - el frontend siempre recibe un informe válido
        return ReportResponse(
            report={
                "error": "No se pudo generar el informe completo.",
                "situation": (
                    "La predicción está disponible, pero el análisis del agente no se "
                    f"pudo completar ({type(exc).__name__})."
                ),
            },
            is_fallback=True,
            warnings=[str(exc)[:200]],
        )
