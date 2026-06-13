"""Utilidades del Model Registry de MLflow: registrar el modelo seleccionado y recargarlo por alias.

Para los modelos delta se versiona el booster interno (flavor nativo) y el run lleva el
tag ``target_representation=delta``; al cargar, se reconstruye el wrapper que devuelve
predicciones en escala de nivel, de modo que API y notebooks consumen siempre la misma
interfaz sin importar la representación del ganador.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.models.train import lag1_feature, setup_mlflow


class _DeltaPredictor:
    """Adaptador de inferencia: suma el lag de sílice al delta predicho."""

    def __init__(self, pyfunc_model, lag_col: str):
        self._model = pyfunc_model
        self._lag_col = lag_col

    def predict(self, X):
        return self._model.predict(X) + X[self._lag_col].to_numpy(dtype=float)


def register_champion(
    config: dict | None = None, family: str = "winner", alias: str = "champion"
) -> dict:
    """Registra el mejor run (por MAE de validación) del ganador y apunta el alias.

    Por defecto busca los runs ``winner`` (los que entrena log_winner_run con los
    parámetros tuneados y el set de features definitivo).
    """
    cfg = config or load_config()
    setup_mlflow(cfg)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.model_family = '{family}'",
        order_by=["metrics.val_mae ASC"],
    )
    if not runs:
        raise RuntimeError(f"No hay ningún run de '{family}' para registrar; ejecuta antes el notebook 02.")
    best = runs[0]
    name = cfg["mlflow"]["registered_model_name"]
    result = mlflow.register_model(f"runs:/{best.info.run_id}/model", name)
    client.set_registered_model_alias(name, alias, result.version)
    return {
        "name": name,
        "version": result.version,
        "alias": alias,
        "family": family,
        "run_id": best.info.run_id,
        "val_mae": best.data.metrics.get("val_mae"),
        "test_mae": best.data.metrics.get("test_mae"),
        "test_r2": best.data.metrics.get("test_r2"),
    }


def champion_uri(config: dict | None = None, alias: str = "champion") -> str:
    cfg = config or load_config()
    return f"models:/{cfg['mlflow']['registered_model_name']}@{alias}"


def load_champion(config: dict | None = None, alias: str = "champion"):
    """Carga el campeón; si fue registrado como delta, devuelve el adaptador a escala nivel."""
    cfg = config or load_config()
    
    db_path = Path(cfg["paths"].get("tracking_db", "mlflow.db"))
    if not db_path.exists():
        return load_packaged_champion(cfg)

    try:
        setup_mlflow(cfg)
        model = mlflow.pyfunc.load_model(champion_uri(cfg, alias))
        run = MlflowClient().get_run(model.metadata.run_id)
        if run.data.tags.get("target_representation") == "delta":
            return _DeltaPredictor(model, lag1_feature(cfg))
        return model
    except Exception:
        return load_packaged_champion(cfg)


def load_packaged_champion(config: dict):
    """Carga el modelo empaquetado para despliegue sin MLflow."""
    out_dir = Path(config["paths"]["models_dir"]) / "champion"
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError("No se encontró MLflow ni artefacto empaquetado en models/champion/")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    family = meta["family"]
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(out_dir / "model.txt"))
        class _LGBWrapper:
            def __init__(self, booster):
                self.booster = booster
            def predict(self, X):
                return self.booster.predict(X)
        base_model = _LGBWrapper(model)
    elif family == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(out_dir / "model.json"))
        class _XGBWrapper:
            def __init__(self, booster):
                self.booster = booster
            def predict(self, X):
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    # Mantener orden de columnas (XGBoost es estricto)
                    feature_names = model.feature_names
                    if feature_names:
                        X = X[feature_names]
                return self.booster.predict(xgb.DMatrix(X))
        base_model = _XGBWrapper(model)
    else:
        raise ValueError(f"Familia no soportada: {family}")
        
    if meta.get("representation") == "delta":
        return _DeltaPredictor(base_model, lag1_feature(config))
    return base_model
