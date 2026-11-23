import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from domain.modelo import predecir

RAIZ = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="API de clasificación de billetes",
    version="1.0.0",
)

# Límite técnico amplio para rechazar entradas absurdas.
Numero = Annotated[
    float,
    Field(ge=-1000, le=1000, allow_inf_nan=False),
]


class Billete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variance: Numero
    skewness: Numero
    curtosis: Numero
    entropy: Numero


class Solicitud(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[Billete] = Field(
        min_length=1,
        max_length=1000,
    )


@lru_cache(maxsize=1)
def obtener_modelo():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            config["mlflow"]["tracking_uri"],
        )
    )

    nombre = config["modelo"]["nombre"]
    alias = config["mlflow"]["alias"]

    try:
        version = MlflowClient().get_model_version_by_alias(
            nombre, alias
        )
        modelo = mlflow.sklearn.load_model(
            f"models:/{nombre}/{version.version}"
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="No se pudo cargar el modelo registrado en MLflow",
        ) from error

    return {
        "modelo": modelo,
        "features": config["datos"]["features"],
        "nombre": nombre,
        "version": version.version,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(recurso=Depends(obtener_modelo)):
    return {
        "status": "ready",
        "modelo": recurso["nombre"],
        "version": recurso["version"],
    }


@app.post("/predict")
def predict(
    solicitud: Solicitud,
    recurso=Depends(obtener_modelo),
):
    datos = pd.DataFrame([
        billete.model_dump()
        for billete in solicitud.instances
    ])

    resultado = predecir(
        recurso["modelo"],
        datos,
        recurso["features"],
    )

    return {
        "modelo": recurso["nombre"],
        "version": recurso["version"],
        "predictions": resultado[
            ["prediction", "probability_class_1"]
        ].to_dict(orient="records"),
    }
