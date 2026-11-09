# Actividad 2 — Producción de petróleo Volve

Prototipo académico para anticipar una caída superior al 20 % de la media de producción en los próximos 30 días frente a los últimos 30. Alcance: registros productores OP y pozos F-12/F-14 con suficiente historial. Champion: regresión logística, versión 1; AUC de validación 0,7015 y prueba 0,4996. No se considera validado para operación.

## Instalación y datos

Desde la raíz del repositorio, con Python 3.13 y Poetry 2:

```powershell
python -m poetry config virtualenvs.in-project true --local
python -m poetry env use python
python -m poetry install
```

Descargar `Volve production data.xlsx` desde el recurso de [Kaggle](https://www.kaggle.com/code/imranulhaquenoor/volve-field-production-data-analysis-eda/input) y colocarlo en `data/`. Se usa la hoja Daily Production Data. El archivo de origen debe obtenerse antes de reproducir los flows; este documento no afirma que el Excel esté incluido en GitHub. Respetar las condiciones de la fuente.

## Servicios

Abrir terminales separadas, todas desde la raíz. En cada terminal de flows, API o deployments definir:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

Terminal MLflow:

```powershell
New-Item -ItemType Directory -Force mlflow_store | Out-Null
python -m poetry run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --artifacts-destination mlflow_store/artifacts --serve-artifacts
```

Terminal Prefect:

```powershell
python -m poetry run prefect server start --host 127.0.0.1 --port 4200
```

Terminal de trabajo, primer entrenamiento e inferencia:

```powershell
python -m poetry run python -m flows.entrenamiento_logistica
python -m poetry run python -m flows.inferencia_produccion
```

El primer entrenamiento que supera el mínimo crea champion. Las siguientes ejecuciones solo lo reemplazan si mejoran también su AUC en las mismas filas de validación. Un empate conserva la versión. Las configuraciones están en `configs/produccion.yaml`, `logistica.yaml`, `features.yaml`, `inferencia.yaml` y `despliegues.yaml`. El run registra features y n_features.

Terminal de deployments, mantener abierta:

```powershell
python -m poetry run python -m flows.desplegar
```

Desde la terminal de trabajo:

```powershell
python -m poetry run prefect deployment run "entrenamiento-logistica-30d/entrenamiento-manual"
python -m poetry run prefect deployment run "inferencia-produccion-30d/inferencia-diaria"
```

La inferencia diaria está programada a las 08:00 America/Bogota. Reproduce un lote histórico del 22/06/2015 y guarda CSV con fecha UTC en `data/predicciones/`. Requiere equipo y procesos activos; no es un servicio alojado permanentemente.

## API y ejemplo

En otra terminal con las variables anteriores:

```powershell
python -m poetry run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger: http://localhost:8000/docs . Salud: GET /health. Predicción: POST /predict. La API recibe variables históricas ya calculadas. Ejemplo reproducible desde PowerShell:

```powershell
$cuerpo = Get-Content -Raw docs/evidencias/actividad-2/api_entrada.json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $cuerpo
```

Resultado observado para F-12: score_caida_30d=0,3075698113, prediccion_caida=0, umbral=0,5, version=1. El score no está calibrado. Pozos fuera del alcance reciben 422. Reiniciar la API después de un cambio de champion.

## Evidencias y límites

[Memoria 2](memoria-2.docx) y [evidencias](evidencias/actividad-2/). La evaluación final del champion está en `evaluacion_prueba_v1.json`. Para reproducirla: `python -m poetry run python -m flows.evaluacion_prueba`; requiere champion y datos. El registro MLflow y los modelos se generan localmente, no se transfieren mediante los enlaces localhost.

Los scripts comparacion_temporal y candidato_individual son exploratorios, no forman parte del deployment principal. No se demostró promoción a una versión 2 mejorada ni reproducción desde clon limpio. La prueba posterior ya fue inspeccionada: los experimentos adicionales no constituyen confirmación independiente. Consultar la memoria para cifras y uso de IA.

