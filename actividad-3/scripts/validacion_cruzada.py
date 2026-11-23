from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

RAIZ = Path(__file__).resolve().parents[1]


def main():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    features = config["datos"]["features"]
    target = config["datos"]["target"]

    parametros = {
        clave: valor
        for clave, valor in config["modelo"].items()
        if clave != "nombre"
    }

    modelo = RandomForestClassifier(**parametros, n_jobs=1)

    particiones = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    resultados = cross_validate(
        modelo,
        train[features],
        train[target],
        cv=particiones,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
        error_score="raise",
    )

    tabla = pd.DataFrame({
        "particion": range(1, 6),
        "accuracy": resultados["test_accuracy"],
        "f1": resultados["test_f1"],
        "roc_auc": resultados["test_roc_auc"],
    })

    print("Resultados por partición:")
    print(tabla.to_string(index=False))

    print("\nMedia y desviación estándar entre particiones:")
    print(
        tabla[["accuracy", "f1", "roc_auc"]]
        .agg(["mean", "std"])
        .to_string()
    )

    destino = RAIZ / "reports" / "validacion_cruzada.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)


if __name__ == "__main__":
    main()