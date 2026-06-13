"""Modelado del delta: predecir el cambio de la sílice en vez del nivel.

Cuando la autocorrelación domina (aquí el lag de 1 h explica la mayor parte de la
varianza), la forma canónica de batir a la persistencia en MAE es convertir el modelo en
un corrector de la persistencia: se entrena sobre y_delta = silica_t - silica_{t-1} y el
nivel se reconstruye sumando el lag conocido. Un modelo delta que predice cero empata
exactamente con la persistencia; cualquier señal real la mejora. La evaluación se hace
siempre en la escala del nivel, comparable con el resto de candidatos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class DeltaTargetRegressor(RegressorMixin, BaseEstimator):
    """Envuelve un regresor de gradient boosting para entrenarlo en escala de cambio.

    `base_model` debe estar ya configurado (parámetros, seed); aquí solo se transforma
    el target al ajustar y se reconstruye el nivel al predecir. El atributo
    `base_model` queda expuesto para SHAP (las atribuciones se interpretan en escala de
    cambio respecto a la hora anterior). Hereda de BaseEstimator/RegressorMixin para
    ser compatible con utilidades de sklearn como permutation_importance.
    """

    def __init__(self, base_model, lag_feature: str):
        self.base_model = base_model
        self.lag_feature = lag_feature

    def _lag(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_feature].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, **fit_params):
        y_delta = np.asarray(y, dtype=float) - self._lag(X)
        if eval_set is not None:
            eval_set = [
                (Xv, np.asarray(yv, dtype=float) - self._lag(Xv)) for Xv, yv in eval_set
            ]
            fit_params["eval_set"] = eval_set
        self.base_model.fit(X, y_delta, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict(X) + self._lag(X)

    def predict_delta(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict(X)
