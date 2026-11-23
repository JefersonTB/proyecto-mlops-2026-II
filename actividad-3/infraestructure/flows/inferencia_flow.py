import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow

from domain.drift import emular_drift
from domain.modelo import predecir, evaluar

RAIZ = Path(__file__).resolve().parents[2]


@flow(name="banknote-inferencia", log_prints=True)
def inferencia_flow():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    features = config["datos"]["features"]
    target = config["datos"]["target"]
    nivel = float(config["inferencia"]["nivel_drift"])

    lote = emular_drift(test, train, config)

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            config["mlflow"]["tracking_uri"],
        )
    )
    mlflow.set_experiment(config["mlflow"]["experimento"])

    nombre = config["modelo"]["nombre"]
    alias = config["mlflow"]["alias"]

    # Resolver el alias y cargar una versión concreta para trazabilidad.
    version = MlflowClient().get_model_version_by_alias(
        nombre, alias
    )
    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}/{version.version}"
    )

    puntuado = predecir(modelo, lote, features)
    metricas = evaluar(modelo, lote, features, target)

    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"{fecha}_{uuid4().hex[:8]}"

    carpeta = RAIZ / "data" / "batches"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_lote = carpeta / f"{batch_id}.csv"
    ruta_metadatos = carpeta / f"{batch_id}.json"

    puntuado.to_csv(ruta_lote, index=False)

    with mlflow.start_run(run_name=f"inferencia-drift-{nivel}") as run:
        metadatos = {
            "batch_id": batch_id,
            "ruta_lote": ruta_lote.relative_to(RAIZ).as_posix(),
            "nivel_drift": nivel,
            "filas": len(puntuado),
            "tasa_clase_1": float(puntuado[target].mean()),
            "modelo": nombre,
            "version_modelo": version.version,
            "run_id": run.info.run_id,
            "metricas": metricas,
        }

        contenido = json.dumps(metadatos, indent=2)
        ruta_metadatos.write_text(contenido, encoding="utf-8")

        mlflow.log_params({
            "nivel_drift": nivel,
            "tamano_lote": len(puntuado),
            "version_modelo": version.version,
            "batch_id": batch_id,
        })
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(ruta_lote), artifact_path="lote")
        mlflow.log_artifact(str(ruta_metadatos), artifact_path="lote")

    # Los detectores podrán localizar la última inferencia exitosa.
    (RAIZ / "data" / "ultimo_lote.json").write_text(
        contenido,
        encoding="utf-8",
    )

    print(contenido)
    return metadatos


if __name__ == "__main__":
    inferencia_flow()