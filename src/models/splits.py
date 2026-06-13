"""Partición cronológica y validación cruzada consciente del tiempo.

Se asume que el frame está ordenado por su DatetimeIndex horario, de modo que el corte
posicional produce una partición train/validación/test estrictamente cronológica, sin
shuffle.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def chronological_split(
    frame: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Las fracciones del split deben sumar 1.")
    n = len(frame)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = frame.iloc[:n_train]
    val = frame.iloc[n_train : n_train + n_val]
    test = frame.iloc[n_train + n_val :]
    return train, val, test


def time_series_cv(n_splits: int) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
