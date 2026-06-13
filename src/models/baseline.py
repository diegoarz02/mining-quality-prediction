"""Baselines ingenuos basados solo en el target.

Ambos baselines reutilizan el valor de laboratorio de la hora anterior, así que asumen
implícitamente que ese resultado ya está disponible. Como las mediciones de laboratorio
llegan con retraso, estos baselines son optimistas; el resultado significativo es un
modelo que los supere. Las predicciones se generan sobre un índice horario completo para
que ningún valor se arrastre a través del hueco de ~13 días.
"""
from __future__ import annotations

import pandas as pd


def to_complete_hourly(target: pd.Series) -> pd.Series:
    full_index = pd.date_range(target.index.min(), target.index.max(), freq="h")
    return target.reindex(full_index)


def persistence_forecast(target: pd.Series) -> pd.Series:
    """y_hat(t) = y(t-1); NaN donde falta la hora anterior (bordes del hueco)."""
    return to_complete_hourly(target).shift(1)


def moving_average_forecast(target: pd.Series, window: int = 3) -> pd.Series:
    """y_hat(t) = media de las `window` horas previas (solo pasado, sin la hora actual)."""
    series = to_complete_hourly(target)
    return series.shift(1).rolling(window).mean()
