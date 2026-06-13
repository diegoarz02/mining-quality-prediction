"""Reproducción de punta a punta desde el CSV crudo.

Construye el dataset horario, entrena y registra todos los modelos en MLflow y registra
el campeón en el Model Registry. Ejecutar desde la raíz del repositorio, con el venv del
proyecto activado:

    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envcheck import ensure_dependencies

ensure_dependencies(["pandas", "numpy", "lightgbm", "xgboost", "sklearn", "mlflow", "yaml"])

from src.config import load_config
from src.data.preprocess import build_processed
from src.models.evaluate import metrics_table
from src.models.registry import register_champion
from src.models.train import log_winner_run, run_experiment


def main() -> None:
    cfg = load_config()
    build_processed(cfg, save=True)
    data = run_experiment(cfg, log=True)
    val_metrics = {name: data["results"][name]["val"] for name in data["results"]}
    test_metrics = {name: data["results"][name]["test"] for name in data["results"]}
    print("\nMétricas de validación (selección):")
    print(metrics_table(val_metrics).round(4).to_string())
    print("\nMétricas de test (reporte final):")
    print(metrics_table(test_metrics).round(4).to_string())
    winner = log_winner_run(cfg)
    print("\nGanador (delta podado):", {k: round(v, 4) for k, v in winner["metrics"]["test"].items()})
    print("\nCampeón registrado:", register_champion(cfg))


if __name__ == "__main__":
    main()
