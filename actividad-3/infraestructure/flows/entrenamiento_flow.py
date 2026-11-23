import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow

from domain.modelo import entrenar, evaluar

RAIZ = Path(__file__).resolve().parents[2]


@flow(name="banknote-entrenamiento", log_prints=True)
def entrenamiento_flow():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    features = config["datos"]["features"]
    target = config["datos"]["target"]

    if set(train["source_id"]) & set(test["source_id"]):
        raise ValueError("Hay registros compartidos entre train y test")

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            config["mlflow"]["tracking_uri"],
        )
    )
    mlflow.set_experiment(config["mlflow"]["experimento"])

    with mlflow.start_run(run_name="entrenamiento-random-forest") as run:
        modelo = entrenar(train, config)
        metricas = evaluar(modelo, test, features, target)

        parametros = {
            clave: valor
            for clave, valor in config["modelo"].items()
            if clave != "nombre"
        }
        mlflow.log_params(parametros)
        mlflow.log_params({
            "filas_entrenamiento": len(train),
            "filas_prueba": len(test),
        })
        mlflow.log_metrics(metricas)

        ejemplo = train[features].head(5)
        firma = infer_signature(
            ejemplo,
            modelo.predict(ejemplo),
        )

        modelo_guardado = mlflow.sklearn.log_model(
            sk_model=modelo,
            name="model",
            signature=firma,
            input_example=ejemplo,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        version = mlflow.register_model(
            model_uri=modelo_guardado.model_uri,
            name=config["modelo"]["nombre"],
        )

        MlflowClient().set_registered_model_alias(
            name=config["modelo"]["nombre"],
            alias=config["mlflow"]["alias"],
            version=version.version,
        )

        resultado = {
            "run_id": run.info.run_id,
            "modelo": config["modelo"]["nombre"],
            "version": version.version,
            "alias": config["mlflow"]["alias"],
            "metricas": metricas,
        }

        destino = RAIZ / "reports" / "entrenamiento.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(resultado, indent=2),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(destino))
        mlflow.log_artifact(str(RAIZ / "configs" / "config.yaml"))

        print(json.dumps(resultado, indent=2))

    return resultado


if __name__ == "__main__":
    entrenamiento_flow()