import os

import mlflow
from prefect import flow, task
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datos_produccion import leer_yaml
from flows.entrenamiento_produccion import ingesta, evaluar, registrar


@task(name="entrenamiento-logistica")
def entrenar_logistica(train, columnas, parametros):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    numericas = [
        c for c in columnas if train[c].dtype.kind in "biuf"
    ]
    categoricas = [c for c in columnas if c not in numericas]
    transformadores = []

    if numericas:
        transformadores.append((
            "numericas",
            Pipeline([
                ("imputar", SimpleImputer(
                    strategy="median", keep_empty_features=True
                )),
                ("escalar", StandardScaler()),
            ]),
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

    modelo = Pipeline([
        ("preprocesar", ColumnTransformer(transformadores)),
        ("clasificar", LogisticRegression(
            C=parametros["C"],
            max_iter=parametros["max_iter"],
            class_weight=parametros["class_weight"],
            random_state=parametros["semilla"],
        )),
    ])
    modelo.fit(train[columnas], train["objetivo"])
    return modelo


@flow(name="entrenamiento-logistica-30d", log_prints=True)
def entrenamiento_logistica():
    cfg = leer_yaml("produccion.yaml")
    parametros = leer_yaml("logistica.yaml")

    # Registrar solo los parametros del algoritmo usado.
    cfg["modelo"] = {
        "nombre": cfg["modelo"]["nombre"],
        "experimento": cfg["modelo"]["experimento"],
        "auc_minimo": cfg["modelo"]["auc_minimo"],
        **parametros,
    }

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(cfg["modelo"]["experimento"])

    train, validacion, columnas = ingesta()
    modelo = entrenar_logistica(train, columnas, parametros)
    metricas = evaluar(modelo, validacion, columnas)
    registrar(modelo, train, validacion, columnas, metricas, cfg)


if __name__ == "__main__":
    entrenamiento_logistica()
