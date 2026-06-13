"""Tests anti-fuga de la ingeniería de features, extendidos a los features nuevos."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.preprocess import load_processed
from src.features.build import build_features, split_xy
from src.models.cv import walk_forward_folds
from src.models.delta import DeltaTargetRegressor


def test_no_leakage_columns_in_features():
    cfg = load_config()
    X, _ = split_xy(build_features(load_processed(cfg), cfg), cfg)
    target = cfg["data"]["target"]
    # El target contemporáneo y la salida del mismo ensayo nunca pueden ser features.
    assert target not in X.columns
    assert "% Iron Concentrate" not in X.columns
    assert "n_samples" not in X.columns
    # Todo feature derivado del target debe ser estrictamente pasado: lags, momentum
    # (diferencia de lags) o rolling de lags. Nada con el valor contemporáneo.
    past_markers = ("__lag", "__mom", "__lagroll")
    for col in X.columns:
        if col.startswith(target):
            assert any(m in col for m in past_markers), col


def test_extended_features_present():
    """Las familias nuevas de features existen cuando sus flags están activos."""
    cfg = load_config()
    X, _ = split_xy(build_features(load_processed(cfg), cfg), cfg)
    assert any("__ewm" in c for c in X.columns)
    assert any("__diff1h" in c for c in X.columns)
    assert any("__slope3h" in c for c in X.columns)
    assert any(c.startswith("cross_air_flow") for c in X.columns)
    assert any(c.startswith("ratio_") for c in X.columns)


def test_features_use_only_past_information():
    """Una fila de features no debe cambiar al añadir datos futuros (sin look-ahead)."""
    cfg = load_config()
    hourly = load_processed(cfg)
    full = build_features(hourly, cfg)
    target = cfg["data"]["target"]

    cut = full.index[1500]
    truncated = build_features(hourly.loc[:cut], cfg)
    common = truncated.index[-1]

    a = full.loc[common].drop(labels=[target]).to_numpy(dtype=float)
    b = truncated.loc[common].drop(labels=[target]).to_numpy(dtype=float)
    # equal_nan: los NaN de calentamiento de los features extendidos deben coincidir.
    assert np.allclose(a, b, equal_nan=True)


def test_delta_regressor_reconstructs_level():
    """Un modelo delta que predice cero debe igualar exactamente a la persistencia."""

    class ZeroModel:
        def fit(self, X, y, **kw):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X = pd.DataFrame({"lag1": [2.0, 3.5, 1.2], "otra": [1.0, 1.0, 1.0]})
    y = pd.Series([2.1, 3.4, 1.5])
    model = DeltaTargetRegressor(ZeroModel(), "lag1").fit(X, y)
    assert np.allclose(model.predict(X), X["lag1"])


def test_walk_forward_folds_respect_time_and_gap():
    """Los folds nunca usan el futuro y la purga separa train de val."""
    n, gap = 1000, 3
    for scheme in ("expanding", "sliding"):
        folds = list(walk_forward_folds(n, n_splits=5, scheme=scheme, gap=gap))
        assert len(folds) == 5
        window = None
        for tr, va in folds:
            assert tr.max() + gap < va.min()
            if scheme == "expanding":
                assert tr.min() == 0
            else:
                # La ventana deslizante mantiene un tamaño de train constante.
                window = window or len(tr)
                assert len(tr) == window
