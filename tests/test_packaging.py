"""Tests del artefacto empaquetado en models/champion/ (modo standalone para Render)."""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pytest

from src.config import load_config
from src.features.build import build_features
from src.models.registry import load_packaged_champion

CFG = load_config()
CHAMP = Path(CFG["paths"]["models_dir"]) / "champion"

pytestmark = pytest.mark.skipif(
    not (CHAMP / "metadata.json").exists(),
    reason="artefacto empaquetado no disponible (ejecutar scripts/package_model.py)",
)


def _meta() -> dict:
    return json.loads((CHAMP / "metadata.json").read_text(encoding="utf-8"))


def test_metadata_carries_real_feature_names():
    meta = _meta()
    assert len(meta["features"]) > 100
    # Nombres reales (con espacios), no los sanitizados por LightGBM en model.txt.
    assert any(" " in name for name in meta["features"])
    booster = lgb.Booster(model_file=str(CHAMP / "model.txt"))
    assert booster.num_feature() == len(meta["features"])


def test_packaged_champion_loads_and_predicts_without_mlflow():
    model = load_packaged_champion(CFG)
    hourly = pd.read_parquet(CHAMP / "sample_hourly.parquet")
    features = _meta()["features"]
    X = build_features(hourly, CFG)[features].dropna()
    preds = model.predict(X.tail(3))
    assert len(preds) == 3
    assert all(0.0 <= float(p) <= 10.0 for p in preds)
