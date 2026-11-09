import json
import os

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datos_produccion import ROOT, leer_yaml, preparar_datos, dividir_datos


def main():
    cfg = leer_yaml("produccion.yaml")
    umbral = leer_yaml("inferencia.yaml")["umbral_clasificacion"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["modelo"]["experimento"])

    nombre = cfg["modelo"]["nombre"]
    version = mlflow.MlflowClient().get_model_version_by_alias(
        nombre, "champion"
    )
    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}/{version.version}"
    )
    codificador = (
        modelo.named_steps["preprocesar"]
        .named_transformers_["categoricas"]
        .named_steps["codificar"]
    )
    pozos = list(codificador.categories_[0])

    prueba = dividir_datos(preparar_datos())["prueba"]
    prueba = prueba.loc[prueba["pozo"].isin(pozos)].copy()
    if prueba["objetivo"].nunique() != 2:
        raise ValueError("La prueba debe contener ambas clases.")

    y = prueba["objetivo"]
    scores = modelo.predict_proba(
        prueba[list(modelo.feature_names_in_)]
    )[:, list(modelo.classes_).index(1)]
    predicciones = (scores >= umbral).astype(int)

    metricas = {
        "auc_prueba": float(roc_auc_score(y, scores)),
        "average_precision_prueba": float(average_precision_score(y, scores)),
        "precision_prueba": float(precision_score(y, predicciones, zero_division=0)),
        "recall_prueba": float(recall_score(y, predicciones, zero_division=0)),
        "proporcion_caidas_prueba": float(y.mean()),
        "n_ejemplos_prueba": len(prueba),
    }

    resumen = {
        "modelo": nombre,
        "version": str(version.version),
        "pozos": pozos,
        "fecha_inicio": str(prueba["fecha"].min().date()),
        "fecha_fin": str(prueba["fecha"].max().date()),
        "umbral_clasificacion": umbral,
        "metricas": metricas,
        "matriz_confusion_filas_real_columnas_prediccion": (
            confusion_matrix(y, predicciones, labels=[0, 1]).tolist()
        ),
    }

    carpeta = ROOT / "docs/evidencias/actividad-2"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"evaluacion_prueba_v{version.version}.json"
    ruta.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with mlflow.start_run(run_name="evaluacion-final-prueba"):
        mlflow.set_tag("fase", "evaluacion_prueba")
        mlflow.log_params({
            "modelo": nombre,
            "version_modelo": version.version,
            "umbral_clasificacion": umbral,
            "pozos": ", ".join(pozos),
        })
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(ruta))
        mlflow.log_artifact(str(ROOT / "flows/evaluacion_prueba.py"))

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print("Evidencia guardada en:", ruta)


if __name__ == "__main__":
    main()
