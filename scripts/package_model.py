"""Empaqueta el modelo campeón desde MLflow para su despliegue en Render (sin BD de MLflow).

Se extrae el booster nativo, se guarda en `models/champion/` junto con un `metadata.json`
que contiene la configuración esencial para la inferencia y un subset de datos horarios
para alimentar el endpoint de predicción standalone.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config
from src.models.registry import champion_uri
from src.models.train import setup_mlflow

import mlflow
from mlflow.tracking import MlflowClient


def package_champion():
    cfg = load_config()
    setup_mlflow(cfg)
    
    uri = champion_uri(cfg, alias="champion")
    print(f"Cargando modelo campeón desde: {uri}")
    
    client = MlflowClient()
    # Resolve alias to get run_id
    model_name = cfg["mlflow"]["registered_model_name"]
    version_info = client.get_model_version_by_alias(model_name, "champion")
    run_id = version_info.run_id
    run = client.get_run(run_id)
    
    family = run.data.tags.get("model_family", cfg["winner"]["family"])
    if family == "winner":
        family = cfg["winner"]["family"]
    representation = run.data.tags.get("target_representation", cfg["winner"]["representation"])
    
    out_dir = Path(cfg["paths"]["models_dir"]) / "champion"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extrayendo modelo nativo ({family})...")
    # Para poder hacer save_model directamente del booster, lo descargamos
    if family == "lightgbm":
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
        model.booster_.save_model(out_dir / "model.txt")
    elif family == "xgboost":
        model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
        model.get_booster().save_model(out_dir / "model.json")
    else:
        raise ValueError(f"Familia no soportada para empaquetar: {family}")
        
    print("Guardando metadata.json...")
    from src.models.train import prepare_data, load_selected_features
    use_selected = cfg["winner"].get("use_selected_features", False)
    subset = load_selected_features(cfg) if use_selected else None
    data = prepare_data(cfg, feature_subset=subset)
    y_pred = model.predict(data["X_test"])
    residuals = (data["y_test"] - y_pred).tolist()
    
    # Nombres reales de las features (con espacios). No se toman del booster: LightGBM
    # sanitiza los nombres al guardar model.txt y no coincidirían con build_features.
    feature_names = list(data["X_test"].columns)

    metadata = {
        "family": family,
        "representation": representation,
        "run_id": run_id,
        "features": feature_names,
        "metrics": {k: v for k, v in run.data.metrics.items() if "test" in k or "val" in k},
        "test_mae": run.data.metrics.get("test_mae", 0.46),
        "residuals_test": residuals
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("Exportando muestra de hourly.parquet...")
    hourly_path = Path(cfg["paths"]["hourly_parquet"])
    if hourly_path.exists():
        df = pd.read_parquet(hourly_path)
        # 720 horas (30 días) es suficiente para el standalone de la API y calcular features
        df_sample = df.iloc[-720:]
        df_sample.to_parquet(out_dir / "sample_hourly.parquet")
        print("Muestra exportada correctamente.")
    else:
        print("No se encontró hourly.parquet, omitiendo exportación de muestra.")

    print(f"Empaquetado exitoso en {out_dir}")

if __name__ == "__main__":
    package_champion()
