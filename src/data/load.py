"""Carga del CSV crudo: parseo de coma decimal, tipado y eliminación de duplicados exactos."""
from __future__ import annotations

import pandas as pd

from src.config import load_config, resolve


def load_raw(config: dict | None = None, csv_path: str | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    path = resolve(csv_path or cfg["paths"]["raw_csv"])
    df = pd.read_csv(
        path,
        decimal=cfg["data"]["decimal"],
        parse_dates=[cfg["data"]["date_col"]],
    )
    return df.sort_values(cfg["data"]["date_col"]).reset_index(drop=True)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina filas totalmente idénticas (duplicados de registro que sesgarían las medias horarias)."""
    before = len(df)
    deduped = df.drop_duplicates(ignore_index=True)
    return deduped, before - len(deduped)
