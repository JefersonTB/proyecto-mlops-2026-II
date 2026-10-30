# Proyecto MLOps — Actividad 1

Repositorio de la actividad de fundamentos, versionamiento y tracking.

## Requisitos
- Python 3.13
- Poetry 2.x
- Git

Los servicios se ejecutan localmente, sin Docker.

## Estructura
- src/: código del proyecto.
- flows/: flow de validación de Prefect y MLflow.
- configs/: configuraciones.
- data/: espacio para datos.
- tests/: espacio para pruebas.
- docs/: memoria y evidencias.

## Instalación
```powershell
git clone https://github.com/JefersonTB/proyecto-mlops-2026-II.git
cd proyecto-mlops-2026-II
python -m pip install poetry
python -m poetry config virtualenvs.in-project true --local
python -m poetry env use python
python -m poetry install
```

Las dependencias están declaradas en pyproject.toml y sus versiones resueltas en poetry.lock.

## Ejecución
Ejecutar los comandos desde la raíz del proyecto, en tres terminales separadas.

### Terminal 1: MLflow
```powershell
New-Item -ItemType Directory -Force mlflow_store | Out-Null
python -m poetry run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --artifacts-destination mlflow_store/artifacts --serve-artifacts
```

### Terminal 2: Prefect
```powershell
python -m poetry run prefect server start --host 127.0.0.1 --port 4200
```

### Terminal 3: validación
```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m poetry run python flows/prueba_flow.py
```

## Resultados esperados
- Prefect: http://localhost:4200, flow prueba en estado Completed.
- Logs: Suma de 1 a 5 = 15.
- MLflow: http://localhost:5000, experimento prueba-validacion.
- Run prueba-mlflow: parámetro n=5 y métrica suma=15.

Mantener abiertas las terminales de los servidores durante la prueba.
Para detenerlos, pulsar Ctrl+C en cada terminal.

## Alcance
Esta entrega valida el entorno y la integración de servicios.
No incluye entrenamiento de un modelo ni un dataset del caso de uso.

## Documentación
- [Memoria](docs/memoria-1.pdf)
- [Evidencias](docs/evidencias/)

## Fuente del flow de prueba
El archivo flows/prueba_flow.py fue tomado del material del curso:
https://github.com/henrycala/santoto-mlops-content/blob/main/actividades/actividad-1/flows/prueba_flow.py

