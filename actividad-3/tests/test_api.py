import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from infraestructure.api.main import app, obtener_modelo


class ModeloFalso:
    classes_ = np.array([0, 1])

    def predict(self, datos):
        return np.zeros(len(datos), dtype=int)

    def predict_proba(self, datos):
        return np.tile([0.8, 0.2], (len(datos), 1))


@pytest.fixture
def cliente():
    def recurso_falso():
        return {
            "modelo": ModeloFalso(),
            "features": [
                "variance",
                "skewness",
                "curtosis",
                "entropy",
            ],
            "nombre": "modelo-prueba",
            "version": "test",
        }

    app.dependency_overrides[obtener_modelo] = recurso_falso

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(obtener_modelo, None)


def billete_valido():
    return {
        "variance": 3.6216,
        "skewness": 8.6661,
        "curtosis": -2.8073,
        "entropy": -0.44699,
    }


def test_health(cliente):
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok"}


def test_ready(cliente):
    respuesta = cliente.get("/ready")

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "status": "ready",
        "modelo": "modelo-prueba",
        "version": "test",
    }


@pytest.mark.parametrize("cantidad", [1, 3])
def test_prediccion_con_modelo_falso(cliente, cantidad):
    respuesta = cliente.post(
        "/predict",
        json={
            "instances": [
                billete_valido() for _ in range(cantidad)
            ]
        },
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()

    assert cuerpo["modelo"] == "modelo-prueba"
    assert cuerpo["version"] == "test"
    assert cuerpo["predictions"] == [
        {
            "prediction": 0,
            "probability_class_1": 0.2,
        }
        for _ in range(cantidad)
    ]


def test_rechaza_campo_faltante(cliente):
    billete = billete_valido()
    del billete["entropy"]

    respuesta = cliente.post(
        "/predict", json={"instances": [billete]}
    )

    assert respuesta.status_code == 422
    errores = respuesta.json()["detail"]
    assert any(
        error["loc"] == ["body", "instances", 0, "entropy"]
        and error["type"] == "missing"
        for error in errores
    )


@pytest.mark.parametrize(
    "valor", [1001, -1001, None, "texto"]
)
def test_rechaza_valores_invalidos(cliente, valor):
    billete = billete_valido()
    billete["variance"] = valor

    respuesta = cliente.post(
        "/predict", json={"instances": [billete]}
    )

    assert respuesta.status_code == 422


def test_rechaza_campo_extra(cliente):
    billete = billete_valido()
    billete["class"] = 0

    respuesta = cliente.post(
        "/predict", json={"instances": [billete]}
    )

    assert respuesta.status_code == 422


@pytest.mark.parametrize("cantidad", [0, 1001])
def test_rechaza_tamano_de_lote_invalido(cliente, cantidad):
    respuesta = cliente.post(
        "/predict",
        json={
            "instances": [
                billete_valido() for _ in range(cantidad)
            ]
        },
    )

    assert respuesta.status_code == 422


def test_respuesta_si_modelo_no_disponible(cliente):
    def recurso_no_disponible():
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible",
        )

    app.dependency_overrides[obtener_modelo] = recurso_no_disponible

    assert cliente.get("/ready").status_code == 503
    assert cliente.post(
        "/predict",
        json={"instances": [billete_valido()]},
    ).status_code == 503

    assert cliente.get("/health").status_code == 200