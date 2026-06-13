"""Entrenamiento de modelos con validación temporal y tracking en MLflow.

El early stopping usa únicamente el split cronológico de validación; el test se toca una
sola vez, para el reporte final, y nunca para tuning ni selección de modelos. Todas las
familias pertenecen al stack de la prueba: LightGBM, XGBoost y scikit-learn.
"""
from __future__ import annotations

import subprocess

import lightgbm as lgb
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_config, resolve
from src.data.preprocess import load_processed
from src.features.build import build_features, load_selected_features, split_xy
from src.models.delta import DeltaTargetRegressor
from src.models.evaluate import regression_metrics
from src.models.splits import chronological_split


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def lag1_feature(cfg: dict) -> str:
    """Nombre del feature de sílice de la hora anterior (ancla del modelo delta)."""
    return f"{cfg['data']['target']}__lag1h"


def prepare_data(config: dict | None = None, feature_subset: list[str] | None = None) -> dict:
    """Construye el frame de features y los splits cronológicos train/val/test.

    ``feature_subset`` restringe las columnas de X (por ejemplo, la lista podada por
    permutation importance) sin alterar las filas ni las fronteras de los splits.
    """
    cfg = config or load_config()
    hourly = load_processed(cfg)
    frame = build_features(hourly, cfg)
    train, val, test = chronological_split(frame, **cfg["split"])
    data = {
        "hourly": hourly,
        "frame": frame,
        "target_range": float(
            train[cfg["data"]["target"]].max() - train[cfg["data"]["target"]].min()
        ),
    }
    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, y = split_xy(part, cfg)
        if feature_subset is not None:
            X = X[feature_subset]
        data[f"X_{name}"], data[f"y_{name}"] = X, y
    data["feature_names"] = list(data["X_train"].columns)
    return data


def make_lgbm(params: dict, seed: int) -> tuple[lgb.LGBMRegressor, int]:
    """Instancia un LGBMRegressor y devuelve (modelo, rondas de early stopping)."""
    p = dict(params)
    stop = p.pop("early_stopping_rounds", 100)
    return lgb.LGBMRegressor(**p, random_state=seed, n_jobs=-1, verbose=-1), stop


def make_xgb(params: dict, seed: int) -> tuple[xgb.XGBRegressor, int]:
    p = dict(params)
    stop = p.pop("early_stopping_rounds", 100)
    model = xgb.XGBRegressor(
        **p, random_state=seed, n_jobs=-1, eval_metric="mae", early_stopping_rounds=stop
    )
    return model, stop


def train_lightgbm(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> lgb.LGBMRegressor:
    model, stop = make_lgbm(params, seed)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stop, verbose=False)],
    )
    return model


def train_xgboost(X_tr, y_tr, X_val, y_val, params: dict, seed: int) -> xgb.XGBRegressor:
    model, _ = make_xgb(params, seed)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm_delta(
    X_tr, y_tr, X_val, y_val, params: dict, seed: int, lag_col: str
) -> DeltaTargetRegressor:
    """LightGBM entrenado sobre el cambio de la sílice (ver src/models/delta.py)."""
    base, stop = make_lgbm(params, seed)
    wrapper = DeltaTargetRegressor(base, lag_col)
    wrapper.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stop, verbose=False)],
    )
    return wrapper


def train_xgboost_delta(
    X_tr, y_tr, X_val, y_val, params: dict, seed: int, lag_col: str
) -> DeltaTargetRegressor:
    base, _ = make_xgb(params, seed)
    wrapper = DeltaTargetRegressor(base, lag_col)
    wrapper.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return wrapper


def train_linear(X_tr, y_tr, kind: str = "ridge") -> Pipeline:
    """Referencia lineal (Ridge o ElasticNet) con imputación y escalado.

    La imputación de mediana cubre los NaN de calentamiento de los features extendidos;
    imputador y escalador se ajustan dentro del pipeline solo con train, de modo que no
    pueden filtrar información de val/test.
    """
    estimator = Ridge(alpha=1.0) if kind == "ridge" else ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ]).fit(X_tr, y_tr)


def train_hgb(X_tr, y_tr, seed: int) -> HistGradientBoostingRegressor:
    """HistGradientBoosting de sklearn como tercera referencia de boosting.

    Sin early stopping interno: su partición de validación es aleatoria (no temporal),
    así que se fija un número de iteraciones moderado.
    """
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed, early_stopping=False,
    )
    return model.fit(X_tr, y_tr)


def train_rf(X_tr, y_tr, seed: int) -> RandomForestRegressor:
    """RandomForest como referencia de bagging (maneja NaN desde sklearn 1.4)."""
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, max_features=0.5,
        random_state=seed, n_jobs=-1,
    )
    return model.fit(X_tr, y_tr)


def _best_iteration(model) -> int | None:
    if isinstance(model, DeltaTargetRegressor):
        model = model.base_model
    if isinstance(model, lgb.LGBMRegressor):
        return model.best_iteration_
    if isinstance(model, xgb.XGBRegressor):
        return int(model.best_iteration)
    return None


def _log_model(model, family: str, input_example: pd.DataFrame) -> None:
    # Para los modelos delta se loguea el booster interno con su flavor nativo; el run
    # lleva el tag target_representation=delta y el registry reconstruye el wrapper.
    inner = model.base_model if isinstance(model, DeltaTargetRegressor) else model
    if family.startswith("lightgbm"):
        mlflow.lightgbm.log_model(inner, name="model", input_example=input_example)
    elif family.startswith("xgboost"):
        mlflow.xgboost.log_model(inner, name="model", input_example=input_example)
    else:
        mlflow.sklearn.log_model(inner, name="model", input_example=input_example)


def evaluate(model, data: dict) -> dict[str, dict]:
    rng = data["target_range"]
    out = {}
    for part in ("val", "test"):
        preds = model.predict(data[f"X_{part}"])
        out[part] = regression_metrics(data[f"y_{part}"], preds, rng)
    return out


def setup_mlflow(config: dict | None = None) -> None:
    cfg = config or load_config()
    # Backend SQLite local (no el file store): soporta el Model Registry y evita el
    # bloqueo por deprecación de MLflow 3. Los artefactos van a mlruns/ vía el
    # artifact_location del experimento, absoluto para no depender del cwd.
    db = resolve(cfg["paths"].get("tracking_db", "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=resolve(cfg["paths"]["mlruns_dir"]).as_uri())
    mlflow.set_experiment(name)


def fit_winner(config: dict | None = None) -> tuple[object, dict]:
    """Ajusta el modelo seleccionado según el bloque ``winner`` del config.

    Lo reutilizan los notebooks 03-05, la API y el grafo de razonamiento, de modo que
    cambiar el ganador en el config propaga el cambio a todo el flujo aguas abajo.
    """
    cfg = config or load_config()
    win = cfg.get("winner", {"family": "lightgbm", "representation": "level",
                             "params_key": "lightgbm", "use_selected_features": False})
    subset = load_selected_features(cfg) if win.get("use_selected_features") else None
    data = prepare_data(cfg, feature_subset=subset)
    params = cfg["models"][win["params_key"]]
    seed = cfg["seed"]
    args = (data["X_train"], data["y_train"], data["X_val"], data["y_val"], params, seed)
    if win["representation"] == "delta":
        trainer = train_lightgbm_delta if win["family"] == "lightgbm" else train_xgboost_delta
        model = trainer(*args, lag_col=lag1_feature(cfg))
    else:
        trainer = train_lightgbm if win["family"] == "lightgbm" else train_xgboost
        model = trainer(*args)
    return model, data


def log_winner_run(config: dict | None = None) -> dict:
    """Entrena el ganador definitivo (bloque ``winner`` del config) y lo loguea en MLflow.

    Es el run que se registra en el Model Registry: usa los parámetros tuneados y, si
    corresponde, el set de features podado. Las familias de run_experiment comparten el
    set completo para que la comparación sea justa; este run es el artefacto final.
    """
    cfg = config or load_config()
    setup_mlflow(cfg)
    win = cfg["winner"]
    model, data = fit_winner(cfg)
    metrics = evaluate(model, data)
    with mlflow.start_run(run_name="winner") as run:
        mlflow.set_tags({
            "model_family": "winner",
            "winner_family": win["family"],
            "target_representation": win["representation"],
            "git_commit": git_commit(),
            "data_version": "hourly-v2-extended",
            "n_features": len(data["feature_names"]),
            "pruned_features": str(bool(win.get("use_selected_features"))),
        })
        mlflow.log_params({f"winner__{k}": v for k, v in cfg["models"][win["params_key"]].items()})
        best_it = _best_iteration(model)
        if best_it is not None:
            mlflow.log_param("winner__best_iteration", best_it)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics["val"].items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics["test"].items()})
        _log_model(model, win["family"], data["X_train"].head(2))
        run_id = run.info.run_id
    return {"run_id": run_id, "metrics": metrics, "n_features": len(data["feature_names"])}


def run_experiment(config: dict | None = None, log: bool = True) -> dict:
    """Entrena las familias base, evalúa en val/test y registra cada run en MLflow."""
    cfg = config or load_config()
    seed = cfg["seed"]
    data = prepare_data(cfg)
    if log:
        setup_mlflow(cfg)

    lag_col = lag1_feature(cfg)
    trainers = {
        "lightgbm": lambda: train_lightgbm(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            cfg["models"]["lightgbm"], seed),
        "xgboost": lambda: train_xgboost(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            cfg["models"]["xgboost"], seed),
        "lightgbm_delta": lambda: train_lightgbm_delta(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            cfg["models"].get("lightgbm_delta", cfg["models"]["lightgbm"]), seed, lag_col),
        "xgboost_delta": lambda: train_xgboost_delta(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            cfg["models"].get("xgboost_delta", cfg["models"]["xgboost"]), seed, lag_col),
        "ridge": lambda: train_linear(data["X_train"], data["y_train"], "ridge"),
    }

    results: dict[str, dict] = {}
    models: dict[str, object] = {}
    run_ids: dict[str, str] = {}
    for family, make in trainers.items():
        model = make()
        metrics = evaluate(model, data)
        models[family] = model
        results[family] = metrics
        if log:
            with mlflow.start_run(run_name=family) as run:
                run_ids[family] = run.info.run_id
                mlflow.set_tags({
                    "model_family": family,
                    "target_representation": "delta" if family.endswith("_delta") else "level",
                    "git_commit": git_commit(),
                    "data_version": "hourly-v2-extended",
                    "n_features": len(data["feature_names"]),
                })
                params = cfg["models"].get(family, cfg["models"].get(family.replace("_delta", ""), {}))
                mlflow.log_params({f"{family}__{k}": v for k, v in params.items()})
                best_it = _best_iteration(model)
                if best_it is not None:
                    mlflow.log_param(f"{family}__best_iteration", best_it)
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics["val"].items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics["test"].items()})
                _log_model(model, family, data["X_train"].head(2))

    data["results"] = results
    data["models"] = models
    data["run_ids"] = run_ids
    return data
