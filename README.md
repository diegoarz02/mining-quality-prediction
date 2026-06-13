# Predicción de calidad en flotación de mineral de hierro

Sistema end-to-end de Data Science y MLOps que anticipa, una hora antes del resultado de laboratorio, el
**% de sílice (impureza) en el concentrado** de una planta de flotación. Cubre el ciclo completo:
comprensión del dato, modelado con validación temporal sin fuga, explicabilidad (SHAP), trazabilidad de
experimentos (MLflow), simulación what-if, un agente de razonamiento (LangGraph + LLM) y una API de
inferencia con interfaz web, lista para desplegar como un solo servicio.

Dataset público: [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) (Kaggle).

## El problema de negocio

En la planta, la calidad del concentrado se confirma por laboratorio con retraso, lo que impide a los
operadores corregir el proceso a tiempo. Anticipar la sílice a nivel horario, con la información
disponible antes del ensayo, habilita decisiones más tempranas y reduce el mineral fuera de
especificación que termina en relaves: un beneficio operativo y ambiental directo.

## Resultados

Selección por MAE de validación; el conjunto de test se evaluó una sola vez con la selección congelada.

| Modelo / baseline | MAE val | MAE test | R² test | MAE test (% rango) |
|---|---|---|---|---|
| Persistencia 1 h (baseline a vencer) | 0.3959 | 0.4727 | 0.5969 | 9.97 |
| Ridge (referencia lineal) | 0.6631 | 0.7417 | 0.3386 | 15.65 |
| LightGBM de nivel (tuneado) | 0.4586 | 0.5201 | 0.6243 | 10.97 |
| **LightGBM delta podado (modelo seleccionado)** | **0.3883** | **0.4599** | **0.6221** | **9.70** |

El modelo seleccionado **supera a la persistencia en MAE y R², en validación y en test**, algo que el
modelo de nivel no lograba. La clave fue predecir el *cambio* respecto a la última medición
(`silica_t − silica_{t-1}`) y reconstruir el nivel sumando el lag conocido: el modelo opera como un
corrector de la persistencia.

## Decisiones técnicas clave

- **Agregación horaria.** El CSV trae ~180 lecturas de sensores por hora y un solo valor de laboratorio
  por hora. Los 19 sensores de alta frecuencia se agregan a media/std/min/max horario; 737k filas crudas
  se convierten en 4097 observaciones reales del target.
- **Exclusión de `% Iron Concentrate` por fuga de datos.** Proviene del mismo ensayo de laboratorio que
  el target (no está disponible al predecir). Incluirla infla artificialmente el desempeño; se excluye.
- **Validación temporal sin fuga.** Split cronológico 70/15/15 sin barajar; toda la selección de
  features, hiperparámetros y modelos ocurre en validación o en *walk-forward* con purga dentro de
  train+val; el test se toca una sola vez al final. El preprocesamiento que aprende parámetros se ajusta
  solo con train.
- **Hallazgo central.** A resolución horaria, la sílice está dominada por su propia autocorrelación; las
  variables de proceso instantáneas aportan señal modesta por sí solas. El valor del modelo crece con el
  retraso del laboratorio: a 6 h de horizonte la persistencia colapsa (R² negativo) mientras el modelo
  mantiene R² positivo. Las variables de proceso afinan la predicción del cambio y conservan valor
  interpretable y de simulación.

## Arquitectura

```
config/        Configuración central (config.yaml): el bloque "winner" propaga a todo el flujo
src/data/      Carga y agregación horaria
src/features/  Ingeniería de features temporal (base + extendidos), sin fuga
src/models/    Splits, baselines, métricas, modelo delta, CV temporal, tuning, ensembles, MLflow, registry
src/explain/   Explicabilidad SHAP
src/simulation/ Escenarios what-if (puntual y de setpoint sostenido)
src/reasoning/ Agente de razonamiento: grafo LangGraph con LLM y fallback determinístico
api/           Servicio FastAPI de inferencia + interfaz web servida en /app
web/           SPA (HTML + CSS + JS vanilla, Chart.js): predicción, SHAP, what-if e informe del agente
notebooks/     01 datos · 02 modelado · 03 SHAP · 04 simulación · 05 razonamiento
scripts/       run_pipeline · tune_models · run_experiments · package_model
tests/         Suite pytest (anti-fuga, CV temporal, delta, API, empaquetado)
models/        Model card, features seleccionadas y artefacto del campeón empaquetado
```

## El modelo (resumen)

LightGBM sobre el delta del target, 110 features (94 podadas por *permutation importance* más las
palancas operativas que alimentan la simulación), hiperparámetros optimizados con Optuna sobre el MAE de
*walk-forward* con purga. Detalle completo en [models/model_card.md](models/model_card.md).

## Explicabilidad y simulación

- **SHAP** global (importancia y *beeswarm*) y local (por predicción): qué empuja la sílice hacia arriba
  o hacia abajo en cada punto de operación. Para el modelo delta, las atribuciones explican el cambio y
  el valor base reconstruye el nivel.
- **What-if** sobre las palancas manipulables (reactivos, flujos de aire, niveles, pH), con efecto
  inmediato y trayectoria de *setpoint* sostenido a 8 h. Las sensibilidades son asociativas, no garantías
  causales de control; el efecto es mayor en las variables que el modelo realmente usa (p. ej. densidad
  de pulpa) y se compone hora a hora.

## Agente de razonamiento (GenAI)

Grafo LangGraph con estado tipado y nodos diferenciados (validación, predicción, *drivers* SHAP,
escenarios, síntesis), con una arista condicional que desvía la entrada inválida a un nodo de
advertencia. El LLM actúa como analista de procesos y solo redacta a partir de cifras ya calculadas; el
nodo de síntesis verifica que todo número del informe exista en el estado del grafo. Sin credenciales de
LLM, el grafo corre en modo determinístico por reglas con el mismo formato de informe (situación,
*drivers*, escenarios, recomendación con impacto, riesgos y límites).

## MLOps y trazabilidad

Backend de MLflow en SQLite con Model Registry: cada técnica del frente de mejora (tuning por familia y
representación, esquemas de CV temporal, corrección de sesgo, poda, ensembles) queda como *run* con su
*tag*. El modelo ganador se registra como `silica-concentrate-regressor@champion`. Para despliegue sin
MLflow, el campeón se exporta a un artefacto nativo autocontenido (`models/champion/`), de modo que el
servicio arranca con carga en tres niveles: registry → artefacto empaquetado → reentrenamiento.

## Cómo ejecutar

Requiere Python 3.13.

```bash
# Entorno
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Coloca el dataset en data/raw/ (ver data/raw/README.md)

# Reproducción end-to-end: datos -> features -> modelos -> MLflow -> registro del campeón
python scripts/run_pipeline.py

# Empaquetar el campeón para despliegue standalone
python scripts/package_model.py

# Servicio + interfaz web (raíz redirige a /app; documentación interactiva en /docs)
python -m uvicorn api.main:app

# Tests
python -m pytest
```

Los notebooks 01 a 05 ejecutan e interpretan el flujo de punta a punta y comparten la lógica de `src/`.

## API

- `POST /predict-from-history` (recomendado): recibe las últimas horas crudas de la planta y construye
  internamente todas las features temporales.
- `POST /predict`: punto de operación parcial; lo no enviado se completa con la referencia.
- `POST /simulate` y `POST /simulate-sustained`: what-if inmediato y trayectoria de setpoint a 8 h.
- `POST /explain`: contribuciones SHAP locales de una predicción.
- `POST /report`: informe del agente de razonamiento (LLM o fallback determinístico).
- `GET /reference`, `GET /health`, `GET /features`: punto de referencia y rangos, estado y catálogo.

Los errores de validación responden en español indicando campo, problema y valor recibido.

## Despliegue (Render)

Despliegue de un solo servicio: el mismo FastAPI sirve la API y la interfaz web, sin CORS ni build de
front separado. El repositorio incluye `render.yaml` (Web Service, build `pip install -r
requirements.txt`, start `uvicorn api.main:app --host 0.0.0.0 --port $PORT`). El servicio carga el modelo
desde el artefacto empaquetado en `models/champion/`, así que funciona sin la base de datos de MLflow.
Para habilitar la narrativa del agente con LLM, se configura la variable de entorno del token en el panel
de Render; sin ella, el informe usa el modo determinístico.

## Limitaciones

- Modelo asociativo: simulaciones direccionales, no garantías de control de proceso.
- A resolución horaria las variables de proceso aportan señal modesta; el modelo se apoya en la dinámica
  reciente del target.
- Subestima parcialmente el régimen de sílice alta (sesgo reducido respecto al modelo de nivel, no
  eliminado).
