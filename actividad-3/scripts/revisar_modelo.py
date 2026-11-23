from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]


def main():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    features = config["datos"]["features"]
    target = config["datos"]["target"]

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    # Comparar características, sin usar IDs ni etiquetas.
    filas_train = set(
        train[features].itertuples(index=False, name=None)
    )
    filas_test = set(
        test[features].itertuples(index=False, name=None)
    )

    print(
        "Combinaciones de features compartidas:",
        len(filas_train & filas_test),
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    nombre = config["modelo"]["nombre"]
    alias = config["mlflow"]["alias"]

    modelo = mlflow.sklearn.load_model(
        f"models:/{nombre}@{alias}"
    )

    print("\nFeatures utilizadas por el modelo:")
    print(list(modelo.feature_names_in_))

    indice_clase_1 = list(modelo.classes_).index(1)

    resultado = test.copy()
    resultado["prediction"] = modelo.predict(test[features])
    resultado["probability_class_1"] = modelo.predict_proba(
        test[features]
    )[:, indice_clase_1]

    matriz = confusion_matrix(
        resultado[target],
        resultado["prediction"],
        labels=[0, 1],
    )

    print("\nMatriz de confusión:")
    print("Filas = clase real; columnas = clase predicha")
    print(pd.DataFrame(
        matriz,
        index=["Real 0", "Real 1"],
        columns=["Predicción 0", "Predicción 1"],
    ))

    errores = resultado[
        resultado[target] != resultado["prediction"]
    ]

    print("\nRegistros mal clasificados:")
    print(errores.to_string(index=False))

    maximo_clase_0 = resultado.loc[
        resultado[target] == 0, "probability_class_1"
    ].max()

    minimo_clase_1 = resultado.loc[
        resultado[target] == 1, "probability_class_1"
    ].min()

    auc = roc_auc_score(
        resultado[target],
        resultado["probability_class_1"],
    )

    print(f"\nAUC: {auc:.10f}")
    print(f"Mayor probabilidad de clase 1 entre los clase 0: {maximo_clase_0:.10f}")
    print(f"Menor probabilidad de clase 1 entre los clase 1: {minimo_clase_1:.10f}")
    print(
        "Separación perfecta de probabilidades:",
        minimo_clase_1 > maximo_clase_0,
    )

    destino = RAIZ / "reports" / "revision_predicciones.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(destino, index=False)


if __name__ == "__main__":
    main()