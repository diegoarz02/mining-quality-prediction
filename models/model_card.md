# Model card: silica-concentrate-regressor

## Resumen
- **Tarea:** regresión de `% Silica Concentrate` a nivel horario en una planta de flotación.
- **Algoritmo:** LightGBM sobre el delta del target (predice `silica_t - silica_{t-1}` y reconstruye el
  nivel sumando el lag conocido), con hiperparámetros de Optuna. Seleccionado por MAE de validación
  frente a XGBoost (delta y nivel), LightGBM de nivel, Ridge y los ensembles blend y stacking.
- **Registro MLflow:** `silica-concentrate-regressor`, alias `champion` (run `winner`).

## Datos de entrenamiento
- 4097 observaciones horarias (marzo a septiembre de 2017), agregadas desde ~737k filas crudas.
- Split cronológico: train 70% (mar-jul), validación 15% (jul-ago), test 15% (ago-sep), sin shuffle.

## Features (110, set podado)
- Base: agregados horarios (media/std/min/max) de 19 sensores, lags 1-3 h y medias móviles 3/6 h de
  sensores y feeds, lags 1-3 h del target, hora del día cíclica.
- Extendidos: suavizados exponenciales (half-life 2/6/12 h), cambios 1 h y pendientes 3 h, dinámica del
  target (momentum, rolling 6/12 h de lags), agregados transversales de las columnas de flotación y
  ratios reactivo/pulpa y aire/pulpa.
- Poda por permutation importance sobre validación (94 features con contribución positiva) más las
  palancas operativas de la hora actual, que son la interfaz de simulación y de la API. Lista completa
  en `models/selected_features.json`.
- **Excluidos por fuga:** `% Iron Concentrate` (mismo ensayo que el target) y el target contemporáneo.

## Desempeño (selección en validación; test tocado una sola vez)
- Validación: MAE 0.3883, R² 0.6289.
- Test: MAE 0.4599, RMSE 0.7366, R² 0.6221, MAE 9.70% del rango.
- Baseline persistencia 1 h: MAE 0.3959 (val) / 0.4727 (test), R² test 0.5969. El modelo la supera en
  MAE y R² en ambos conjuntos.
- Estado anterior del proyecto (LightGBM de nivel): MAE test 0.563, R² 0.603; la mejora es de 18% en
  MAE.

## Supuestos y uso previsto
- Asume que la sílice de horas previas está disponible al predecir (retraso de laboratorio ~1 h).
- Pensado como soporte a la decisión (alerta temprana de calidad), no como lazo de control automático.

## Limitaciones
- Modelo asociativo: no implica causalidad; las simulaciones son direccionales.
- Subestima el régimen de sílice alta (sesgo de tercil alto ~+0.4 en test, reducido desde +0.82 del
  modelo de nivel).
- Las variables de proceso instantáneas aportan poca señal a resolución horaria; el modelo corrige la
  persistencia con la dinámica reciente del target y las usa para afinar el cambio.

## Reproducción
`python scripts/run_pipeline.py` reconstruye datos, reentrena las familias, loguea en MLflow, entrena
el ganador y re-registra el alias `champion`. El tuning se reproduce con `python scripts/tune_models.py`
y los experimentos de mejora con `python scripts/run_experiments.py`.
