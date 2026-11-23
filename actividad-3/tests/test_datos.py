from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from domain.modelo import validar_datos

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture
def datos():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    return train, test, config


def test_esquema_y_valores_validos(datos):
    train, test, config = datos
    features = config["datos"]["features"]
    target = config["datos"]["target"]

    for df in [train, test]:
        validar_datos(df, features, target)

        assert set(df.columns) == set(
            ["source_id", target] + features
        )
        assert all(
            pd.api.types.is_numeric_dtype(df[columna])
            for columna in features
        )
        assert set(df[target]) == {0, 1}


def test_rangos_de_features(datos):
    train, test, config = datos
    features = config["datos"]["features"]

    # Contrato técnico amplio, coherente con la API.
    for df in [train, test]:
        assert df[features].ge(-1000).all().all()
        assert df[features].le(1000).all().all()


def test_separacion_sin_coincidencias(datos):
    train, test, config = datos
    features = config["datos"]["features"]

    assert train["source_id"].is_unique
    assert test["source_id"].is_unique
    assert set(train["source_id"]).isdisjoint(test["source_id"])

    filas_train = set(
        train[features].itertuples(index=False, name=None)
    )
    filas_test = set(
        test[features].itertuples(index=False, name=None)
    )
    assert filas_train.isdisjoint(filas_test)


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_rechaza_nulos_e_infinitos(datos, valor):
    train, _, config = datos
    features = config["datos"]["features"]
    target = config["datos"]["target"]

    corrupto = train.copy()
    corrupto.loc[corrupto.index[0], features[0]] = valor

    with pytest.raises(ValueError, match="nulos o infinitos"):
        validar_datos(corrupto, features, target)


def test_rechaza_columna_faltante(datos):
    train, _, config = datos
    features = config["datos"]["features"]

    incompleto = train.drop(columns=features[0])

    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_datos(
            incompleto, features, config["datos"]["target"]
        )


def test_rechaza_target_invalido(datos):
    train, _, config = datos
    target = config["datos"]["target"]

    corrupto = train.copy()
    corrupto.loc[corrupto.index[0], target] = 2

    with pytest.raises(ValueError, match="solo 0 y 1"):
        validar_datos(
            corrupto, config["datos"]["features"], target
        )


def test_rechaza_datos_vacios(datos):
    train, _, config = datos

    with pytest.raises(ValueError, match="vacío"):
        validar_datos(
            train.iloc[:0],
            config["datos"]["features"],
            config["datos"]["target"],
        )