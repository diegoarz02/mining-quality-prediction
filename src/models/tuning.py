"""Búsqueda de hiperparámetros con Optuna (TPE) sobre walk-forward temporal.

El objetivo es el MAE medio (escala de nivel) en los folds walk-forward con purga sobre
train+val. El test no participa: la selección completa ocurre dentro de train+val. Cada
estudio se registra en MLflow como un run con el tag ``technique`` y la tabla completa
de trials como artefacto.
"""
from __future__ import annotations

import optuna
import pandas as pd

from src.models.cv import cross_validate_temporal
from src.models.train import (
    train_lightgbm,
    train_lightgbm_delta,
    train_xgboost,
    train_xgboost_delta,
)

N_ESTIMATORS = 2500
EARLY_STOP = 100


def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def make_builder(family: str, representation: str, params: dict, seed: int, lag_col: str):
    """Devuelve un callable (X_tr, y_tr, X_va, y_va) -> modelo ajustado en escala nivel."""
    if family == "lightgbm" and representation == "delta":
        return lambda Xt, yt, Xv, yv: train_lightgbm_delta(Xt, yt, Xv, yv, params, seed, lag_col)
    if family == "lightgbm":
        return lambda Xt, yt, Xv, yv: train_lightgbm(Xt, yt, Xv, yv, params, seed)
    if family == "xgboost" and representation == "delta":
        return lambda Xt, yt, Xv, yv: train_xgboost_delta(Xt, yt, Xv, yv, params, seed, lag_col)
    if family == "xgboost":
        return lambda Xt, yt, Xv, yv: train_xgboost(Xt, yt, Xv, yv, params, seed)
    raise ValueError(f"Familia desconocida: {family}")


def tune_model(
    family: str,
    representation: str,
    X: pd.DataFrame,
    y: pd.Series,
    lag_col: str,
    seed: int,
    n_trials: int = 100,
    n_splits: int = 5,
    gap: int = 3,
    scheme: str = "expanding",
) -> optuna.Study:
    """Estudio TPE; el objetivo es el MAE medio walk-forward (con purga) en train+val."""
    space = _lgbm_space if family == "lightgbm" else _xgb_space

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        builder = make_builder(family, representation, params, seed, lag_col)
        folds = cross_validate_temporal(builder, X, y, n_splits=n_splits, scheme=scheme, gap=gap)
        trial.set_user_attr("mae_std", float(folds["mae"].std()))
        return float(folds["mae"].mean())

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize", sampler=sampler,
        study_name=f"{family}-{representation}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def study_params(study: optuna.Study) -> dict:
    """Parámetros completos del mejor trial, listos para el bloque models del config."""
    params = dict(study.best_params)
    params["n_estimators"] = N_ESTIMATORS
    params["early_stopping_rounds"] = EARLY_STOP
    if "lightgbm" in study.study_name:
        params["subsample_freq"] = 1
    return params
