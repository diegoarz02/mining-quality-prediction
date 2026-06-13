import numpy as np
import pandas as pd

from src.config import load_config
from src.data.load import drop_exact_duplicates
from src.data.preprocess import aggregate_hourly


def _synthetic_raw(cfg: dict, n_per_hour: int = 180, hours: int = 3) -> pd.DataFrame:
    cols = (
        cfg["data"]["feed_cols"]
        + cfg["data"]["sensor_cols"]
        + cfg["data"]["leakage_excluded"]
        + [cfg["data"]["target"]]
    )
    rng = np.random.default_rng(0)
    rows = []
    for h in range(hours):
        ts = pd.Timestamp("2017-04-01") + pd.Timedelta(hours=h)
        for _ in range(n_per_hour):
            row = {c: float(rng.normal(100, 5)) for c in cols}
            row[cfg["data"]["date_col"]] = ts
            rows.append(row)
    return pd.DataFrame(rows)


def test_drop_exact_duplicates_removes_repeats():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": [3.0, 3.0, 4.0]})
    deduped, removed = drop_exact_duplicates(df)
    assert removed == 1
    assert len(deduped) == 2


def test_aggregate_hourly_is_clean_and_typed():
    cfg = load_config()
    hourly = aggregate_hourly(_synthetic_raw(cfg), cfg)
    target = cfg["data"]["target"]
    assert len(hourly) == 3
    assert target in hourly.columns
    assert int(hourly.isna().sum().sum()) == 0
    # Cada sensor aporta las cuatro agregaciones.
    for agg in ("mean", "std", "min", "max"):
        assert any(c.endswith(f"__{agg}") for c in hourly.columns)
    # El número de muestras por hora queda registrado para auditoría de calidad de datos.
    assert (hourly["n_samples"] == 180).all()
