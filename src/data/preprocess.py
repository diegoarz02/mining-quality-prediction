"""Agregación horaria y manejo de huecos.

La planta registra ~180 muestras de sensores por hora (cada ~20 s) mientras el target
de laboratorio se reporta por hora. Agregar los sensores de alta frecuencia a
estadísticas horarias convierte cada fila en una observación real del target y elimina
la replicación artificial de 180x.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config, resolve
from src.data.load import drop_exact_duplicates, load_raw


def aggregate_hourly(df_raw: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    date_col = cfg["data"]["date_col"]
    target = cfg["data"]["target"]
    sensors = cfg["data"]["sensor_cols"]
    feeds = cfg["data"]["feed_cols"]
    aggs = cfg["data"]["agg_funcs"]

    grouped = df_raw.groupby(date_col)

    sensor_agg = grouped[sensors].agg(aggs)
    sensor_agg.columns = [f"{col}__{func}" for col, func in sensor_agg.columns]

    # Unas pocas horas tienen ~180 muestras byte-idénticas (sensor congelado o registro
    # repetido); tras deduplicar colapsan a una sola fila y su std queda indefinida. La
    # dispersión intra-hora observada en ese caso es cero. Es un relleno constante: no
    # aprende nada de los datos y no puede filtrar información entre train y test.
    std_cols = [c for c in sensor_agg.columns if c.endswith("__std")]
    sensor_agg[std_cols] = sensor_agg[std_cols].fillna(0.0)

    # Los feeds son constantes dentro de la hora: la media reproduce el valor de
    # laboratorio exactamente.
    feed_agg = grouped[feeds].mean()

    # La variación intra-hora del target es despreciable (corr(media, primero) = 0.998);
    # la media es el representante horario más estable.
    target_agg = grouped[target].mean().rename(target)

    n_samples = grouped.size().rename("n_samples")

    # Las salidas del mismo ensayo de laboratorio se conservan (media horaria) solo para
    # que el notebook 02 cuantifique la fuga que introducen. La construcción de features
    # las descarta por diseño.
    leak_cols = cfg["data"].get("leakage_excluded", [])
    parts = [feed_agg, sensor_agg]
    if leak_cols:
        parts.append(grouped[leak_cols].mean())
    parts += [target_agg, n_samples]

    hourly = pd.concat(parts, axis=1)
    hourly.index.name = date_col
    return hourly.sort_index()


def hourly_gaps(hourly: pd.DataFrame) -> pd.DataFrame:
    """Devuelve los huecos contiguos de la línea de tiempo horaria (más de una hora entre filas)."""
    idx = pd.Series(hourly.index)
    diffs = idx.diff()
    gap_mask = diffs > pd.Timedelta(hours=1)
    gaps = pd.DataFrame(
        {
            "gap_start": idx[gap_mask].values - diffs[gap_mask].values + pd.Timedelta(hours=1),
            "gap_end": idx[gap_mask].values,
            "missing_hours": (diffs[gap_mask].dt.total_seconds() / 3600 - 1).astype(int).values,
        }
    )
    return gaps.reset_index(drop=True)


def build_processed(config: dict | None = None, save: bool = True) -> pd.DataFrame:
    """Pipeline completo crudo -> horario. Devuelve el frame horario y opcionalmente lo cachea."""
    cfg = config or load_config()
    raw = load_raw(cfg)
    raw, _ = drop_exact_duplicates(raw)
    hourly = aggregate_hourly(raw, cfg)
    if save:
        out = resolve(cfg["paths"]["hourly_parquet"])
        out.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_parquet(out)
    return hourly


def load_processed(config: dict | None = None) -> pd.DataFrame:
    """Carga el frame horario cacheado, construyéndolo en el primer uso."""
    cfg = config or load_config()
    path = resolve(cfg["paths"]["hourly_parquet"])
    if path.exists():
        return pd.read_parquet(path)
    return build_processed(cfg, save=True)
