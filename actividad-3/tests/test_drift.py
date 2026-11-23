from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from domain.drift import (
    emular_drift,
    detectar_drift_target,
    detectar_drift_features,
)

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recursos():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    return train, test, config


def crear_lote(recursos, nivel):
    train, test, config = recursos
    escenario = deepcopy(config)
    escenario["inferencia"]["nivel_drift"] = nivel

    return emular_drift(test, train, escenario)


@pytest.mark.parametrize("nivel", [0.0, 0.3, 0.6])
def test_emulacion_conserva_origen_y_etiquetas(recursos, nivel):
    train, test, config = recursos
    original = test.copy(deep=True)
    lote = crear_lote(recursos, nivel)
    target = config["datos"]["target"]

    assert len(lote) == config["inferencia"]["tamano_lote"]
    assert lote["source_id"].is_unique
    assert set(lote["source_id"]).issubset(test["source_id"])
    assert set(lote["source_id"]).isdisjoint(train["source_id"])

    etiquetas = test.set_index("source_id").loc[
        lote["source_id"], target
    ]

    assert (lote[target].to_numpy() == etiquetas.to_numpy()).all()

    pd.testing.assert_frame_equal(test, original)
    pd.testing.assert_frame_equal(
        lote, crear_lote(recursos, nivel)
    )


def test_nivel_cero_conserva_features(recursos):
    _, test, config = recursos
    features = config["datos"]["features"]
    lote = crear_lote(recursos, 0.0)

    origen = test.set_index("source_id").loc[
        lote["source_id"], features
    ]

    pd.testing.assert_frame_equal(
        lote[features].reset_index(drop=True),
        origen.reset_index(drop=True),
    )


@pytest.mark.parametrize("nivel", [-0.1, 1.1, float("nan")])
def test_rechaza_niveles_invalidos(recursos, nivel):
    with pytest.raises(ValueError, match="nivel de drift"):
        crear_lote(recursos, nivel)


@pytest.mark.parametrize(
    "nivel,esperada",
    [(0.0, False), (0.3, False), (0.6, True)],
)
def test_alerta_target(recursos, nivel, esperada):
    train, _, config = recursos
    lote = crear_lote(recursos, nivel)

    resultado = detectar_drift_target(train, lote, config)

    assert resultado["alerta"] == esperada


def test_target_rechaza_lote_sin_etiquetas(recursos):
    train, _, config = recursos
    lote = crear_lote(recursos, 0.6)
    lote = lote.drop(columns=config["datos"]["target"])

    with pytest.raises(ValueError, match="etiquetas reales"):
        detectar_drift_target(train, lote, config)


@pytest.mark.parametrize(
    "nivel,esperada",
    [(0.0, False), (0.3, True), (0.6, True)],
)
def test_alerta_features_y_reporte(recursos, nivel, esperada, tmp_path):
    train, _, config = recursos
    lote = crear_lote(recursos, nivel)

    resultado, snapshot = detectar_drift_features(
        train, lote, config
    )

    assert resultado["alerta"] == esperada
    assert set(resultado["columnas"]) == set(
        config["datos"]["features"]
    )

    if nivel == 0.6:
        assert resultado["columnas"]["variance"]["drift"]
        assert resultado["columnas"]["entropy"]["drift"]

    destino = tmp_path / "features.html"
    snapshot.save_html(str(destino))
    assert destino.is_file()
    assert destino.stat().st_size > 1000