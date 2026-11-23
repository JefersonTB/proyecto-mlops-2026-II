import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def validar_datos(df, features, target=None):
    columnas = features + ([target] if target else [])

    faltantes = set(columnas) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    if df.empty:
        raise ValueError("El conjunto de datos está vacío")

    valores = df[columnas].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Hay valores nulos o infinitos")

    if target and not df[target].isin([0, 1]).all():
        raise ValueError("La variable objetivo debe contener solo 0 y 1")


def entrenar(train, config):
    features = config["datos"]["features"]
    target = config["datos"]["target"]
    parametros = config["modelo"]

    validar_datos(train, features, target)

    if train[target].nunique() != 2:
        raise ValueError("El entrenamiento requiere ambas clases")

    modelo = RandomForestClassifier(
        n_estimators=parametros["n_estimators"],
        max_depth=parametros["max_depth"],
        min_samples_leaf=parametros["min_samples_leaf"],
        random_state=parametros["random_state"],
        n_jobs=1,
    )

    modelo.fit(train[features], train[target])
    return modelo


def predecir(modelo, datos, features):
    validar_datos(datos, features)

    resultado = datos.copy()
    resultado["prediction"] = modelo.predict(datos[features])

    indice_clase_1 = list(modelo.classes_).index(1)
    resultado["probability_class_1"] = modelo.predict_proba(
        datos[features]
    )[:, indice_clase_1]

    return resultado


def evaluar(modelo, datos, features, target):
    validar_datos(datos, features, target)
    resultado = predecir(modelo, datos, features)

    metricas = {
        "accuracy": float(
            accuracy_score(datos[target], resultado["prediction"])
        ),
        "f1_class_1": float(
            f1_score(
                datos[target],
                resultado["prediction"],
                zero_division=0,
            )
        ),
    }

    # ROC AUC necesita que existan ambas clases en el lote.
    if datos[target].nunique() == 2:
        metricas["roc_auc"] = float(
            roc_auc_score(
                datos[target],
                resultado["probability_class_1"],
            )
        )

    return metricas