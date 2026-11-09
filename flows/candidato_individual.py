import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from flows.comparacion_temporal import construir
from flows.entrenamiento_produccion import ingesta
from src.datos_produccion import ROOT, leer_yaml


def predecir(paquete, datos):
    desconocidos = set(datos["pozo"]) - set(paquete["modelos"])
    if desconocidos:
        raise ValueError(f"Pozos fuera del alcance: {desconocidos}")

    scores = np.empty(len(datos))
    for pozo, modelo in paquete["modelos"].items():
        posiciones = np.flatnonzero(datos["pozo"].eq(pozo).to_numpy())
        if len(posiciones):
            indice = list(modelo.classes_).index(1)
            scores[posiciones] = modelo.predict_proba(
                datos.iloc[posiciones][paquete["variables"]]
            )[:, indice]
    return scores


def main():
    cfg = leer_yaml("produccion.yaml")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["modelo"]["experimento"])

    train, validacion, columnas = ingesta.fn()
    variables = [c for c in columnas if c != "pozo"]
    paquete = {
        "modelos": {},
        "variables": variables,
        "horizonte_dias": cfg["objetivo"]["horizonte_dias"],
        "C": 0.01,
    }

    for pozo, grupo in train.groupby("pozo"):
        if grupo["objetivo"].nunique() != 2:
            raise ValueError(f"Entrenamiento sin ambas clases: {pozo}")
        modelo = construir("logistica_0.01", variables)
        modelo.fit(grupo[variables], grupo["objetivo"])
        paquete["modelos"][pozo] = modelo

    scores = predecir(paquete, validacion)
    metricas = {
        "auc_validacion": float(
            roc_auc_score(validacion["objetivo"], scores)
        ),
        "average_precision_validacion": float(
            average_precision_score(validacion["objetivo"], scores)
        ),
    }

    with tempfile.TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "modelos_por_pozo.joblib"
        joblib.dump(paquete, ruta)

        # Se carga exclusivamente el archivo creado por este proceso.
        recuperado = joblib.load(ruta)
        np.testing.assert_allclose(
            scores, predecir(recuperado, validacion),
            rtol=0, atol=0,
        )

        with mlflow.start_run(run_name="candidato-individual-C001") as run:
            mlflow.set_tags({
                "fase": "entrenamiento",
                "estado": "candidato_exploratorio",
                "decision": "sin_promocion",
                "serializacion_verificada": "true",
            })
            mlflow.log_params({
                "algoritmo": "regresion_logistica_por_pozo",
                "C": 0.01,
                "class_weight": "balanced",
                "features": leer_yaml("features.yaml")["seleccion"],
                "n_features_por_modelo": len(variables),
                "pozos": ", ".join(sorted(paquete["modelos"])),
                "filas_entrenamiento": len(train),
                "filas_validacion": len(validacion),
            })
            mlflow.log_metrics(metricas)
            mlflow.log_dict(
                {"variables": variables, "enrutamiento": "pozo"},
                "variables.json",
            )
            mlflow.log_artifact(str(ruta), artifact_path="paquete")
            for archivo in [
                "configs/produccion.yaml",
                "configs/features.yaml",
                "src/datos_produccion.py",
                "flows/comparacion_temporal.py",
                "flows/entrenamiento_produccion.py",
                "flows/candidato_individual.py",
                "docs/evidencias/actividad-2/comparacion_temporal.csv",
            ]:
                mlflow.log_artifact(str(ROOT / archivo))

            mlflow.set_tag(
                "dataset_sha256",
                hashlib.sha256(
                    (ROOT / cfg["datos"]["archivo"]).read_bytes()
                ).hexdigest(),
            )
            print("Run del candidato:", run.info.run_id)

    print("Metricas:", metricas)
    print("OK: scores identicos despues de guardar y cargar.")
    print("Paquete guardado como artefacto en MLflow.")
    print("Champion y API sin modificaciones.")


if __name__ == "__main__":
    main()
