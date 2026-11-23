from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from domain.modelo import entrenar, evaluar, predecir

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recursos():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    modelo = entrenar(train, config)
    return modelo, train, test, config


def test_modelo_utiliza_solo_features(recursos):
    modelo, _, _, config = recursos

    assert list(modelo.feature_names_in_) == config["datos"]["features"]
    assert set(modelo.classes_) == {0, 1}


def test_calidad_minima_en_prueba(recursos):
    modelo, _, test, config = recursos

    metricas = evaluar(
        modelo,
        test,
        config["datos"]["features"],
        config["datos"]["target"],
    )

    # Umbrales de regresión: detectar deterioros importantes.
    assert metricas["accuracy"] >= 0.90
    assert metricas["f1_class_1"] >= 0.90
    assert metricas["roc_auc"] >= 0.95


def test_predicciones_validas_y_datos_preservados(recursos):
    modelo, _, test, config = recursos
    original = test.copy(deep=True)

    resultado = predecir(
        modelo, test, config["datos"]["features"]
    )

    assert len(resultado) == len(test)
    assert set(resultado["prediction"]).issubset({0, 1})
    assert resultado["probability_class_1"].between(0, 1).all()

    pd.testing.assert_frame_equal(test, original)
    pd.testing.assert_frame_equal(resultado[test.columns], original)


def test_inferencia_individual_coincide_con_lote(recursos):
    modelo, _, test, config = recursos
    features = config["datos"]["features"]
    muestra = test.head(5)

    conjunto = predecir(modelo, muestra, features)

    for posicion in range(len(muestra)):
        individual = predecir(
            modelo, muestra.iloc[[posicion]], features
        )

        assert (
            individual["prediction"].iloc[0]
            == conjunto["prediction"].iloc[posicion]
        )
        assert np.isclose(
            individual["probability_class_1"].iloc[0],
            conjunto["probability_class_1"].iloc[posicion],
        )


def test_entrenamiento_reproducible(recursos):
    modelo, train, test, config = recursos
    segundo_modelo = entrenar(train, config)
    features = config["datos"]["features"]

    np.testing.assert_allclose(
        modelo.predict_proba(test[features]),
        segundo_modelo.predict_proba(test[features]),
        rtol=0,
        atol=1e-12,
    )


def test_rechaza_entrenamiento_con_una_clase(recursos):
    _, train, _, config = recursos
    target = config["datos"]["target"]
    una_clase = train[train[target] == 0]

    with pytest.raises(ValueError, match="ambas clases"):
        entrenar(una_clase, config)