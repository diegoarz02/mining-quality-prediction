"""Ingeniería de features temporal.

Cada feature usa o bien los sensores de proceso de la hora actual (disponibles en tiempo
real) o valores estrictamente pasados. El target de laboratorio nunca se usa de forma
contemporánea (llega con retraso) y las salidas del mismo ensayo se descartan. Los lags y
ventanas se calculan sobre un índice horario completo, de modo que nunca puentean el hueco
de ~13 días de los datos.

Los features extendidos (EWM, diferencias, agregados transversales, ratios, dinámica del
target) se añaden sin alterar el conjunto de filas del frame base: sus NaN de
calentamiento se conservan porque LightGBM y XGBoost los manejan de forma nativa y los
modelos lineales imputan con la mediana de train. Así los splits cronológicos son
idénticos entre el modelo base y el extendido y las comparaciones siguen siendo válidas.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve


def _excluded(cfg: dict) -> set[str]:
    return {cfg["data"]["target"], "n_samples", *cfg["data"].get("leakage_excluded", [])}


def base_feature_columns(hourly: pd.DataFrame, cfg: dict) -> list[str]:
    excl = _excluded(cfg)
    return [c for c in hourly.columns if c not in excl]


def _lag_roll_columns(hourly: pd.DataFrame, cfg: dict) -> list[str]:
    """Medias horarias de sensores y feeds de laboratorio: las señales cuya historia
    reciente vale la pena seguir."""
    feeds = [c for c in cfg["data"]["feed_cols"] if c in hourly.columns]
    means = [c for c in hourly.columns if c.endswith("__mean")]
    return means + feeds


def _core_features(h: pd.DataFrame, hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Set base de features (idéntico al modelo original): hora actual, lags y rolling
    de sensores, lags del target y hora del día."""
    target = cfg["data"]["target"]
    fcfg = cfg["features"]

    frame = h[base_feature_columns(hourly, cfg)].copy()

    history_cols = _lag_roll_columns(hourly, cfg)
    for lag in fcfg["sensor_lags"]:
        lagged = h[history_cols].shift(lag)
        lagged.columns = [f"{c}__lag{lag}h" for c in history_cols]
        frame = pd.concat([frame, lagged], axis=1)
    for window in fcfg["rolling_windows"]:
        rolled = h[history_cols].rolling(window).mean()
        rolled.columns = [f"{c}__roll{window}h_mean" for c in history_cols]
        frame = pd.concat([frame, rolled], axis=1)

    if fcfg.get("use_target_lags", False):
        # Sílice de horas estrictamente pasadas, disponible bajo el retraso de
        # laboratorio asumido (~1 h).
        for lag in fcfg.get("target_lags", []):
            frame[f"{target}__lag{lag}h"] = h[target].shift(lag)

    if fcfg.get("add_time_features", True):
        hour = frame.index.hour
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return frame


def _extended_features(h: pd.DataFrame, hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Features extendidos del Frente A. Solo usan la hora actual o el pasado."""
    target = cfg["data"]["target"]
    fcfg = cfg["features"]
    history_cols = _lag_roll_columns(hourly, cfg)
    parts: list[pd.DataFrame] = []

    # Suavizado exponencial de sensores y feeds. Los sensores de la hora actual están
    # disponibles en tiempo real, así que el EWM puede incluirla. El hueco de ~13 días
    # no transmite información: con half-life <= 12 h, el peso de lo anterior al hueco
    # decae a 2^(-318/12) ~ 0.
    for hl in fcfg.get("ewm_halflives", []):
        ewm = h[history_cols].ewm(halflife=hl).mean()
        ewm.columns = [f"{c}__ewm{hl}h" for c in history_cols]
        parts.append(ewm)

    # Cambio respecto a la hora anterior y pendiente de las últimas 3 h.
    if fcfg.get("add_diffs", False):
        diff1 = h[history_cols].diff(1)
        diff1.columns = [f"{c}__diff1h" for c in history_cols]
        slope3 = (h[history_cols] - h[history_cols].shift(3)) / 3.0
        slope3.columns = [f"{c}__slope3h" for c in history_cols]
        parts += [diff1, slope3]

    # Agregados transversales de las 7 columnas de flotación: la planta las opera en
    # conjunto y el detalle individual puede ser ruido.
    if fcfg.get("add_cross_aggregates", False):
        cross = pd.DataFrame(index=h.index)
        for group, label in [("Air Flow", "air_flow"), ("Level", "level")]:
            cols = [c for c in hourly.columns if group in c and c.endswith("__mean")]
            block = h[cols]
            cross[f"cross_{label}_mean"] = block.mean(axis=1)
            cross[f"cross_{label}_std"] = block.std(axis=1)
            cross[f"cross_{label}_min"] = block.min(axis=1)
            cross[f"cross_{label}_max"] = block.max(axis=1)
        parts.append(cross)

    # Ratios con sentido metalúrgico: dosis de reactivos y aire por unidad de pulpa.
    if fcfg.get("add_ratios", False):
        pulp = h["Ore Pulp Flow__mean"]
        air_cols = [c for c in hourly.columns if "Air Flow" in c and c.endswith("__mean")]
        ratios = pd.DataFrame(index=h.index)
        ratios["ratio_amina_pulp"] = h["Amina Flow__mean"] / pulp
        ratios["ratio_starch_pulp"] = h["Starch Flow__mean"] / pulp
        ratios["ratio_air_total_pulp"] = h[air_cols].sum(axis=1) / pulp
        parts.append(ratios)

    # Dinámica del target con valores estrictamente pasados (shift >= 1): momentum de
    # la última hora conocida y nivel/dispersión recientes.
    if fcfg.get("use_target_lags", False) and fcfg.get("add_target_dynamics", False):
        past = h[target].shift(1)
        dyn = pd.DataFrame(index=h.index)
        dyn[f"{target}__mom1h"] = h[target].shift(1) - h[target].shift(2)
        for w in fcfg.get("target_rolling_windows", []):
            dyn[f"{target}__lagroll{w}h_mean"] = past.rolling(w).mean()
            dyn[f"{target}__lagroll{w}h_std"] = past.rolling(w).std()
        parts.append(dyn)

    if not parts:
        return pd.DataFrame(index=h.index)
    return pd.concat(parts, axis=1)


def _assemble(hourly: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Construye el frame completo de features sobre un índice horario continuo.

    Devuelve (frame, columnas del set base) para que cada consumidor decida qué filas
    conservar.
    """
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    h = hourly.reindex(full_index)
    h.index.name = hourly.index.name

    core = _core_features(h, hourly, cfg)
    extended = _extended_features(h, hourly, cfg)
    frame = pd.concat([core, extended], axis=1)
    frame["__target__"] = h[cfg["data"]["target"]]
    return frame, list(core.columns)


def build_features(hourly: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    target = cfg["data"]["target"]
    frame, core_cols = _assemble(hourly, cfg)
    frame = frame.rename(columns={"__target__": target})
    # Se descartan filas del hueco, calentamiento de lags/rolling del set base y horas
    # sin target. Los NaN de calentamiento de los features extendidos se conservan para
    # no alterar el conjunto de filas (ver docstring del módulo).
    return frame.dropna(subset=[*core_cols, target]).copy()


def build_inference_features(hourly: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Versión para servir: no exige target en la última hora ni descarta filas.

    La usa el endpoint /predict-from-history de la API: la hora a predecir tiene
    sensores pero aún no tiene resultado de laboratorio. Los NaN que queden (historia
    corta) se completan aguas arriba con el punto de referencia.
    """
    cfg = config or load_config()
    frame, _ = _assemble(hourly, cfg)
    return frame.drop(columns=["__target__"])


def load_selected_features(config: dict | None = None) -> list[str] | None:
    """Lee la lista de features podada por permutation importance, si existe."""
    cfg = config or load_config()
    rel = cfg["features"].get("selected_features_file")
    if not rel:
        return None
    path = resolve(rel)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def split_xy(frame: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    cfg = config or load_config()
    target = cfg["data"]["target"]
    return frame.drop(columns=[target]), frame[target]
