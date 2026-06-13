"""Esquemas de solicitud y respuesta del servicio de inferencia.

Los nombres de campos JSON se mantienen en inglés (contrato técnico); todas las
descripciones visibles en /docs están en español.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from api.examples import HISTORY_EXAMPLE, PREDICT_EXAMPLE, SIMULATE_EXAMPLE


class PredictRequest(BaseModel):
    """Punto de operación parcial: solo las variables que se conocen.

    Las features no enviadas se completan con el punto de operación de referencia
    (mediana de los últimos 30 días de entrenamiento), de modo que no hace falta armar
    el vector completo a mano. Para una predicción con historia real, usar
    /predict-from-history.
    """

    features: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Pares {nombre_de_feature: valor}. Acepta cualquier subconjunto de las "
            "features del modelo (ver GET /features); el resto se completa con la "
            "referencia. Ejemplo de nombres: '% Silica Concentrate__lag1h' (sílice de "
            "la hora anterior), 'Ore Pulp pH__mean' (pH medio de la hora actual)."
        ),
    )

    model_config = {"json_schema_extra": {"examples": [PREDICT_EXAMPLE]}}


class PredictResponse(BaseModel):
    predicted_silica: float = Field(description="Sílice del concentrado predicha para la hora (%).")
    n_features_provided: int = Field(description="Cantidad de features enviadas por el cliente.")
    n_features_filled: int = Field(description="Cantidad de features completadas con la referencia.")


class HistoryRecord(BaseModel):
    """Una hora de operación de la planta, con las 21 variables base del proceso."""

    date: datetime = Field(description="Marca de tiempo de la hora (se trunca a la hora en punto).")
    values: dict[str, float] = Field(
        description=(
            "Variables de proceso de esa hora: '% Iron Feed', '% Silica Feed' y las 19 "
            "medias horarias de sensores con su nombre crudo (por ejemplo 'Starch Flow', "
            "'Ore Pulp pH', 'Flotation Column 01 Air Flow')."
        ),
    )
    silica: float | None = Field(
        default=None,
        description=(
            "Resultado de laboratorio de % de sílice de esa hora. Obligatorio para todas "
            "las horas excepto la última, que es la que se predice (su laboratorio aún "
            "no está disponible)."
        ),
    )


class PredictFromHistoryRequest(BaseModel):
    """Las últimas N horas crudas de la planta; el servicio construye las features.

    Es el modo recomendado: los lags, medias móviles y suavizados se calculan sobre la
    historia real entregada, en lugar de completarse con la referencia. Se recomienda
    enviar 24 horas; el mínimo es 3 (con historia corta, las features de ventanas
    largas se completan con la referencia).
    """

    history: list[HistoryRecord] = Field(
        min_length=3,
        description="Horas ordenadas cronológicamente; la última es la hora a predecir.",
    )

    model_config = {"json_schema_extra": {"examples": [HISTORY_EXAMPLE]}}


class PredictFromHistoryResponse(BaseModel):
    predicted_silica: float = Field(description="Sílice del concentrado predicha para la última hora (%).")
    target_hour: datetime = Field(description="Hora a la que corresponde la predicción.")
    n_hours_received: int = Field(description="Horas de historia recibidas.")
    n_features_computed: int = Field(description="Features calculadas desde la historia entregada.")
    n_features_filled: int = Field(description="Features completadas con la referencia (historia corta).")


class SimulateRequest(BaseModel):
    """Escenario what-if: cambios aditivos sobre un punto de operación base."""

    base_features: dict[str, float] = Field(
        default_factory=dict,
        description="Punto de operación base (parcial, igual que en /predict).",
    )
    deltas: dict[str, float] = Field(
        ...,
        description=(
            "Cambios aditivos por feature, por ejemplo {'Amina Flow__mean': 50} para "
            "simular +50 unidades de flujo de amina."
        ),
    )

    model_config = {"json_schema_extra": {"examples": [SIMULATE_EXAMPLE]}}


class SimulateResponse(BaseModel):
    base_silica: float = Field(description="Sílice predicha en el punto de operación base (%).")
    simulated_silica: float = Field(description="Sílice predicha tras aplicar los cambios (%).")
    delta_silica: float = Field(description="Efecto estimado del escenario (puntos de sílice).")
    applied: dict[str, float] = Field(description="Valor final de cada variable modificada.")
    out_of_observed_range: list[str] = Field(
        description="Variables que quedaron fuera del rango histórico observado (extrapolación)."
    )


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' si el servicio está listo para predecir.")
    model_source: str = Field(
        description="Origen del modelo: 'mlflow_registry' (alias champion) o 'trained_fallback'."
    )
    n_features: int = Field(description="Cantidad de features que espera el modelo.")


class FeaturesResponse(BaseModel):
    n_features: int = Field(description="Cantidad total de features del modelo.")
    features: list[str] = Field(description="Nombres de las features, en el orden del modelo.")


class ReferenceResponse(BaseModel):
    reference: dict[str, float] = Field(description="Punto de operación de referencia (mediana histórica).")
    min: dict[str, float] = Field(description="Valores mínimos históricos por variable.")
    max: dict[str, float] = Field(description="Valores máximos históricos por variable.")
    p5: dict[str, float] = Field(description="Percentil 5 histórico por variable.")
    p95: dict[str, float] = Field(description="Percentil 95 histórico por variable.")
    mae_test: float = Field(description="Error absoluto medio (MAE) en el conjunto de prueba, útil para bandas de confianza.")
    residuals_test: list[float] = Field(description="Muestra de residuos empíricos en test, útil para calcular empíricamente la probabilidad de exceder un umbral asumiendo que los errores futuros se distribuyen igual que en test.")


class ExplainContribution(BaseModel):
    feature: str = Field(description="Nombre crudo de la columna (e.g., 'Ore Pulp pH__mean').")
    label: str = Field(description="Etiqueta legible para la planta (e.g., 'Ore Pulp pH (mean)').")
    value: float = Field(description="Valor de la variable en la predicción actual.")
    shap_value: float = Field(description="Contribución SHAP, con signo (dirección) y magnitud.")


class ExplainResponse(BaseModel):
    predicted_silica: float = Field(description="Predicción total de sílice (%).")
    base_value: float = Field(description="Valor base esperado (baseline del modelo).")
    contributions: list[ExplainContribution] = Field(description="Top contribuciones SHAP ordenadas por magnitud.")


class ReportRequest(BaseModel):
    features: dict[str, float] = Field(
        default_factory=dict, 
        description="Punto de operación actual (parcial o completo)."
    )


class ReportResponse(BaseModel):
    report: dict = Field(description="Informe estructurado del agente LangGraph (narrativa, factores clave, recomendaciones).")
    is_fallback: bool = Field(description="True si se usó el modo determinístico por falla/falta de LLM.")
    warnings: list[str] = Field(default_factory=list, description="Advertencias durante la generación del reporte.")


class SimulateSustainedRequest(BaseModel):
    base_features: dict[str, float] = Field(
        default_factory=dict,
        description="Punto de operación base (parcial, igual que en /predict)."
    )
    deltas: dict[str, float] = Field(
        ...,
        description="Cambios aditivos por feature para mantener de forma sostenida."
    )


class SimulateSustainedResponse(BaseModel):
    trajectory: list[float] = Field(description="Predicciones de sílice para cada hora simulada con el escenario.")
    trajectory_base: list[float] = Field(description="Trayectoria base esperada (inercial) sin aplicar el delta.")
    delta_accumulated: float = Field(default=0.0, description="Diferencia acumulada estimada en todo el horizonte.")
