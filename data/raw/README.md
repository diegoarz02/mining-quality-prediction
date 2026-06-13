# Datos crudos

Este directorio no se versiona (ver `.gitignore`). Para reproducir desde cero, coloca aquí el archivo:

```
data/raw/MiningProcess_Flotation_Plant_Database.csv
```

## Descarga

Dataset: *Quality Prediction in a Mining Process* (Kaggle).

```bash
kaggle datasets download -d edumagalhaes/quality-prediction-in-a-mining-process
unzip quality-prediction-in-a-mining-process.zip -d data/raw/
```

Requiere credenciales de Kaggle en `~/.kaggle/kaggle.json`. Alternativamente, descarga el CSV manualmente desde la página del dataset y cópialo en esta carpeta.

El archivo usa coma como separador decimal y comillas en los valores numéricos; la carga lo maneja en `src/data/load.py`.
