import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.datos_produccion import leer_yaml


class EntradaPozo(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pozo: str
    tendencia_7d_30d: float | None = Field(ge=0)
    ratio_produccion_mes_anterior: float | None = Field(ge=0)
    variabilidad_relativa: float | None = Field(ge=0)
    fraccion_ceros_30d: float | None = Field(ge=0, le=1)
    horas_media_30d: float | None = Field(ge=0, le=24)
    cambio_horas_7d_30d: float | None = Field(ge=-24, le=24)
    corte_agua_30d: float | None = Field(ge=0, le=1)
    cambio_corte_agua_30d: float | None = Field(ge=-1, le=1)
    ratio_presion_7d_30d: float | None = Field(ge=0)
    cambio_choke_7d_30d: float | None


@asynccontextmanager
async def lifespan(app):
    cfg = leer_yaml("produccion.yaml")
    config_inferencia = leer_yaml("inferencia.yaml")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    nombre = cfg["modelo"]["nombre"]
    version = mlflow.MlflowClient().get_model_version_by_alias(
        nombre, "champion"
    )
    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}/{version.version}"
    )

    columnas = list(modelo.feature_names_in_)
    if set(columnas) != set(EntradaPozo.model_fields):
        raise RuntimeError(
            "Las variables del champion no coinciden con el esquema de la API."
        )

    codificador = (
        modelo.named_steps["preprocesar"]
        .named_transformers_["categoricas"]
        .named_steps["codificar"]
    )
    app.state.modelo = modelo
    app.state.columnas = columnas
    app.state.pozos = list(codificador.categories_[0])
    app.state.version = str(version.version)
    app.state.nombre = nombre
    app.state.umbral = config_inferencia["umbral_clasificacion"]
    yield


app = FastAPI(
    title="Volve: pronostico de caida de produccion a 30 dias",
    description=(
        "Prototipo academico. Recibe variables historicas ya calculadas. "
        "El score no es una probabilidad calibrada."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": app.state.nombre,
        "version": app.state.version,
        "pozos_admitidos": app.state.pozos,
    }


@app.post("/predict")
def predict(entrada: EntradaPozo):
    if entrada.pozo not in app.state.pozos:
        raise HTTPException(
            status_code=422,
            detail="Pozo fuera del alcance del modelo.",
        )

    fila = pd.DataFrame([entrada.model_dump()])
    for columna in app.state.columnas:
        if columna != "pozo":
            fila[columna] = pd.to_numeric(fila[columna], errors="raise")

    score = float(
        app.state.modelo.predict_proba(
            fila[app.state.columnas]
        )[0, 1]
    )
    return {
        "pozo": entrada.pozo,
        "score_caida_30d": score,
        "prediccion_caida": int(score >= app.state.umbral),
        "umbral": app.state.umbral,
        "modelo": app.state.nombre,
        "version": app.state.version,
    }
