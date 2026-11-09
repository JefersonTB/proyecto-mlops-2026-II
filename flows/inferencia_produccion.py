import os
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task

from src.datos_produccion import ROOT, leer_yaml, preparar_datos


@task(name="cargar-champion")
def cargar_champion(nombre):
    cliente = mlflow.MlflowClient()
    version = cliente.get_model_version_by_alias(nombre, "champion")

    # Resolver el alias en cada ejecucion y fijar esa version durante el lote.
    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}/{version.version}"
    )

    # El alcance se obtiene del modelo entrenado, no de una lista manual.
    codificador = (
        modelo.named_steps["preprocesar"]
        .named_transformers_["categoricas"]
        .named_steps["codificar"]
    )
    pozos = list(codificador.categories_[0])
    print("Champion:", nombre, "version:", version.version)
    print("Pozos admitidos:", pozos)
    return modelo, str(version.version), pozos


@task(name="preparar-lote")
def preparar_lote(pozos, config):
    datos = preparar_datos()
    corte = pd.Timestamp(config["fecha_corte"])

    # Exigir datos exactamente en la fecha indicada evita usar datos obsoletos.
    lote = datos.loc[
        datos["pozo"].isin(pozos) & datos["fecha"].eq(corte)
    ].sort_values("pozo").head(config["tamano_lote"]).copy()

    if lote.empty:
        raise ValueError("No hay datos validos para el corte y pozos elegidos.")

    # La etiqueta real no participa en la inferencia.
    return lote.drop(columns=["objetivo"], errors="ignore")


@task(name="predecir")
def predecir(modelo, lote, config):
    columnas = list(modelo.feature_names_in_)
    probabilidades = modelo.predict_proba(lote[columnas])[:, 1]

    resultado = lote[["fecha", "pozo"]].copy()
    resultado["score_caida_30d"] = probabilidades
    resultado["prediccion_caida"] = (
        probabilidades >= config["umbral_clasificacion"]
    ).astype(int)
    return resultado


@task(name="guardar-predicciones")
def guardar(resultado, version, cfg, config):
    instante = datetime.now(timezone.utc)
    salida = ROOT / config["carpeta_salida"]
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / (
        "predicciones_" + instante.strftime("%Y%m%dT%H%M%S%fZ") + ".csv"
    )

    resultado["version_modelo"] = version
    resultado["ejecucion_utc"] = instante.isoformat()
    resultado["modo"] = "simulacion_historica"
    resultado.to_csv(ruta, index=False, encoding="utf-8")

    with mlflow.start_run(run_name="inferencia-volve-30d"):
        mlflow.set_tags({
            "fase": "inferencia",
            "modo": "simulacion_historica",
        })
        mlflow.log_params({
            "modelo": cfg["modelo"]["nombre"],
            "alias": "champion",
            "version_modelo": version,
            "fecha_corte": config["fecha_corte"],
            "umbral_clasificacion": config["umbral_clasificacion"],
            "tamano_lote_configurado": config["tamano_lote"],
        })
        mlflow.log_metric("n_predicciones", len(resultado))
        mlflow.log_artifact(str(ruta), artifact_path="predicciones")
        mlflow.log_artifact(str(ROOT / "configs/inferencia.yaml"))
        mlflow.log_artifact(str(ROOT / "flows/inferencia_produccion.py"))

    print(resultado.to_string(index=False))
    print("CSV guardado en:", ruta)
    return str(ruta)


@flow(name="inferencia-produccion-30d", log_prints=True)
def inferencia_produccion():
    cfg = leer_yaml("produccion.yaml")
    config = leer_yaml("inferencia.yaml")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["modelo"]["experimento"])

    modelo, version, pozos = cargar_champion(cfg["modelo"]["nombre"])
    lote = preparar_lote(pozos, config)
    resultado = predecir(modelo, lote, config)
    return guardar(resultado, version, cfg, config)


if __name__ == "__main__":
    inferencia_produccion()
