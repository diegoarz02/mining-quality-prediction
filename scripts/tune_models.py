"""Tuning sistemático con Optuna para las cuatro combinaciones familia x representación.

Ejecutar desde la raíz del repo con el venv del proyecto activado:

    python scripts/tune_models.py

La selección usa exclusivamente walk-forward con purga dentro de train+val; el test no
se toca. Cada estudio queda en MLflow (tag ``technique``) con la tabla de trials como
artefacto, y los mejores parámetros se guardan en reports/tuning/ para copiarlos al
bloque ``models`` del config.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import optuna
import pandas as pd

from src.config import load_config, resolve, set_seeds
from src.models.train import git_commit, lag1_feature, prepare_data, setup_mlflow
from src.models.tuning import study_params, tune_model

# (familia, representación, n_trials): más presupuesto donde está la apuesta principal.
STUDIES = [
    ("lightgbm", "delta", 120),
    ("xgboost", "delta", 80),
    ("lightgbm", "level", 60),
    ("xgboost", "level", 60),
]


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config()
    set_seeds(cfg["seed"])
    data = prepare_data(cfg)
    # Train + val concatenados: el walk-forward interno define sus propios folds.
    X = pd.concat([data["X_train"], data["X_val"]])
    y = pd.concat([data["y_train"], data["y_val"]])
    lag_col = lag1_feature(cfg)
    n_splits = cfg["cv"]["n_splits"]
    gap = cfg["cv"]["gap_hours"]

    out_dir = resolve("reports/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(cfg)

    for family, representation, n_trials in STUDIES:
        name = f"{family}-{representation}"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Iniciando estudio {name} ({n_trials} trials)...", flush=True)
        study = tune_model(
            family, representation, X, y, lag_col, cfg["seed"],
            n_trials=n_trials, n_splits=n_splits, gap=gap, scheme="expanding",
        )
        best = study_params(study)
        elapsed = (time.time() - t0) / 60

        trials = study.trials_dataframe()
        trials_path = out_dir / f"trials_{name}.csv"
        trials.to_csv(trials_path, index=False)
        with open(out_dir / f"best_{name}.json", "w", encoding="utf-8") as fh:
            json.dump({"family": family, "representation": representation,
                       "cv_mae_mean": study.best_value,
                       "cv_mae_std": study.best_trial.user_attrs.get("mae_std"),
                       "params": best}, fh, indent=2)

        with mlflow.start_run(run_name=f"optuna-{name}"):
            mlflow.set_tags({
                "technique": f"optuna-{family}-{representation}",
                "experiment_type": "tuning",
                "git_commit": git_commit(),
                "cv_scheme": f"expanding+gap{gap}",
                "n_trials": n_trials,
            })
            mlflow.log_params({f"best__{k}": v for k, v in best.items()})
            mlflow.log_metrics({
                "cv_mae_mean": study.best_value,
                "cv_mae_std": study.best_trial.user_attrs.get("mae_std", float("nan")),
            })
            mlflow.log_artifact(str(trials_path))

        print(f"[{time.strftime('%H:%M:%S')}] {name}: CV MAE {study.best_value:.4f} "
              f"({elapsed:.1f} min)", flush=True)

    print("Tuning completo.", flush=True)


if __name__ == "__main__":
    main()
