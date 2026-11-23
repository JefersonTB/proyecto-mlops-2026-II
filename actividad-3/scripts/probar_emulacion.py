from pathlib import Path

import pandas as pd
import yaml

from domain.drift import emular_drift

RAIZ = Path(__file__).resolve().parents[1]


def main():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    train = pd.read_csv(RAIZ / config["datos"]["entrenamiento"])
    test = pd.read_csv(RAIZ / config["datos"]["prueba"])

    features = config["datos"]["features"]
    target = config["datos"]["target"]

    # La función trabaja sobre una copia de prueba.
    test_original = test.copy(deep=True)
    originales = test.set_index("source_id")

    resumen = []

    for nivel in [0.0, 0.3, 0.6]:
        config["inferencia"]["nivel_drift"] = nivel
        lote = emular_drift(test, train, config)

        assert len(lote) == config["inferencia"]["tamano_lote"]
        assert lote["source_id"].is_unique
        assert set(lote["source_id"]).issubset(set(test["source_id"]))
        assert set(lote["source_id"]).isdisjoint(set(train["source_id"]))

        origen = originales.loc[lote["source_id"]]

        assert (
            lote[target].to_numpy() == origen[target].to_numpy()
        ).all(), "Se modificaron etiquetas reales"

        if nivel == 0.0:
            pd.testing.assert_frame_equal(
                lote[features].reset_index(drop=True),
                origen[features].reset_index(drop=True),
            )

        # Verificar que repetir la operación produce el mismo lote.
        repetido = emular_drift(test, train, config)
        pd.testing.assert_frame_equal(lote, repetido)

        resumen.append({
            "nivel": nivel,
            "filas": len(lote),
            "tasa_clase_1": lote[target].mean(),
            "cambio_medio_variance": (
                lote["variance"].to_numpy()
                - origen["variance"].to_numpy()
            ).mean(),
            "cambio_medio_entropy": (
                lote["entropy"].to_numpy()
                - origen["entropy"].to_numpy()
            ).mean(),
        })

    pd.testing.assert_frame_equal(test, test_original)

    tabla = pd.DataFrame(resumen)
    print(tabla.to_string(index=False))
    print("\nTodas las comprobaciones pasaron.")

    destino = RAIZ / "reports" / "prueba_emulacion.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(destino, index=False)


if __name__ == "__main__":
    main()