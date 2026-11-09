import hashlib
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.datos_produccion import (
    ROOT, dividir_datos, leer_yaml, preparar_datos, seleccionar_variables,
)


@task(name="ingesta")
def ingesta():
    cfg = leer_yaml("produccion.yaml")
    datos = preparar_datos()
    columnas = seleccionar_variables(datos)
    partes = dividir_datos(datos)
    train = partes["entrenamiento"]
    validacion = partes["validacion"]

    resumen = train.groupby("pozo")["objetivo"].agg(
        ejemplos="size", caidas="sum"
    )
    resumen["sin_caida"] = resumen["ejemplos"] - resumen["caidas"]
    criterio = cfg["elegibilidad"]

    elegibles = resumen.index[
        (resumen["ejemplos"] >= criterio["min_ejemplos"])
        & (resumen["caidas"] >= criterio["min_caidas"])
        & (resumen["sin_caida"] >= criterio["min_sin_caida"])
    ].tolist()

    if not elegibles:
        raise ValueError("Ningun pozo cumple el criterio de historial.")

    train = train.loc[train["pozo"].isin(elegibles)].copy()
    validacion = validacion.loc[
        validacion["pozo"].isin(elegibles)
    ].copy()

    if train["objetivo"].nunique() != 2:
        raise ValueError("Entrenamiento sin ambas clases.")
    if validacion["objetivo"].nunique() != 2:
        raise ValueError("Validacion sin ambas clases.")

    print("Pozos elegibles:", elegibles)
    print("Entrenamiento:", len(train))
    print("Validacion:", len(validacion))
    return train, validacion, columnas


@task(name="entrenamiento")
def entrenar(train, columnas, cfg):
    numericas = [
        c for c in columnas
        if pd.api.types.is_numeric_dtype(train[c])
    ]
    categoricas = [c for c in columnas if c not in numericas]
    transformadores = []

    if numericas:
        transformadores.append((
            "numericas",
            SimpleImputer(strategy="median", keep_empty_features=True),
            numericas,
        ))
    if categoricas:
        transformadores.append((
            "categoricas",
            Pipeline([
                ("imputar", SimpleImputer(strategy="most_frequent")),
                ("codificar", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categoricas,
        ))

    parametros = cfg["modelo"]
    modelo = Pipeline([
        ("preprocesar", ColumnTransformer(transformadores)),
        ("clasificar", RandomForestClassifier(
            n_estimators=parametros["n_estimators"],
            max_depth=parametros["max_depth"],
            min_samples_leaf=parametros["min_samples_leaf"],
            class_weight=parametros["class_weight"],
            random_state=parametros["semilla"],
            n_jobs=-1,
        )),
    ])
    modelo.fit(train[columnas], train["objetivo"])
    return modelo


@task(name="evaluacion")
def evaluar(modelo, validacion, columnas):
    probabilidades = modelo.predict_proba(validacion[columnas])[:, 1]
    metricas = {
        "auc_validacion": float(
            roc_auc_score(validacion["objetivo"], probabilidades)
        ),
        "average_precision_validacion": float(
            average_precision_score(validacion["objetivo"], probabilidades)
        ),
    }
    print("Metricas:", metricas)
    return metricas


@task(name="registro")
def registrar(modelo, train, validacion, columnas, metricas, cfg):
    cliente = MlflowClient()
    nombre = cfg["modelo"]["nombre"]
    auc_champion = None
    version_actual = None

    try:
        registrado = cliente.get_registered_model(nombre)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        version_actual = registrado.aliases.get("champion")

    with mlflow.start_run(run_name="entrenamiento-volve-30d") as run:
        mlflow.set_tags({
            "fase": "entrenamiento",
            "objetivo": "caida_produccion_30d",
        })
        mlflow.log_params({
            **cfg["modelo"],
            **cfg["objetivo"],
            **cfg["fechas"],
            "features": leer_yaml("features.yaml")["seleccion"],
            "n_features": len(columnas),
            "filas_entrenamiento": len(train),
            "filas_validacion": len(validacion),
        })
        mlflow.log_metrics(metricas)
        mlflow.log_dict({"variables": columnas}, "variables.json")
        mlflow.log_artifact(str(ROOT / "configs/produccion.yaml"))
        mlflow.log_artifact(str(ROOT / "configs/features.yaml"))
        mlflow.log_artifact(str(ROOT / "src/datos_produccion.py"))
        mlflow.log_artifact(str(ROOT / "flows/entrenamiento_produccion.py"))

        ruta = ROOT / cfg["datos"]["archivo"]
        mlflow.set_tag(
            "dataset_sha256",
            hashlib.sha256(ruta.read_bytes()).hexdigest(),
        )

        # Evaluar al champion sobre las mismas filas que al candidato.
        if version_actual is not None:
            champion = mlflow.sklearn.load_model(
                f"models:/{nombre}/{version_actual}"
            )
            variables_champion = list(champion.feature_names_in_)
            probabilidades = champion.predict_proba(
                validacion[variables_champion]
            )[:, 1]
            auc_champion = float(
                roc_auc_score(validacion["objetivo"], probabilidades)
            )
            mlflow.log_metric("auc_champion_validacion", auc_champion)
            mlflow.set_tag("champion_anterior", str(version_actual))

        auc = metricas["auc_validacion"]
        promover = (
            auc > cfg["modelo"]["auc_minimo"]
            and (auc_champion is None or auc > auc_champion)
        )

        # Guardar el candidato en el run no implica crear una version.
        ejemplo = train[columnas].head(5)
        info = mlflow.sklearn.log_model(
            sk_model=modelo,
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            name="modelo",
            input_example=ejemplo,
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
        )

        if promover:
            version = mlflow.register_model(info.model_uri, nombre)
            cliente.set_registered_model_alias(
                nombre, "champion", version.version
            )
            mlflow.set_tag("decision", "promovido")
            mlflow.set_tag("version_promovida", str(version.version))
            print("CHAMPION actualizado a version", version.version)
        else:
            mlflow.set_tag("decision", "no_promovido")
            print("Candidato guardado; champion sin cambios.")
            if version_actual is None:
                print("Todavia no hay champion: no se supero el AUC minimo.")

        print("AUC candidato:", auc)
        print("AUC champion anterior:", auc_champion)
        print("Run MLflow:", run.info.run_id)


@flow(name="entrenamiento-produccion-30d", log_prints=True)
def entrenamiento_produccion():
    cfg = leer_yaml("produccion.yaml")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["modelo"]["experimento"])

    train, validacion, columnas = ingesta()
    modelo = entrenar(train, columnas, cfg)
    metricas = evaluar(modelo, validacion, columnas)
    registrar(modelo, train, validacion, columnas, metricas, cfg)


if __name__ == "__main__":
    entrenamiento_produccion()
