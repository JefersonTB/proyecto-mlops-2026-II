import json
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, get_run_logger

from domain.drift import detectar_drift_features

RAIZ = Path(__file__).resolve().parents[2]


@flow(name="banknote-drift-features", log_prints=True)
def drift_features_flow():
    logger = get_run_logger()

    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    with (RAIZ / "data" / "ultimo_lote.json").open(
        encoding="utf-8"
    ) as archivo:
        metadatos = json.load(archivo)

    referencia = pd.read_csv(
        RAIZ / config["datos"]["entrenamiento"]
    )
    lote = pd.read_csv(RAIZ / metadatos["ruta_lote"])

    resultado, snapshot = detectar_drift_features(
        referencia,
        lote,
        config,
    )

    resultado["batch_id"] = metadatos["batch_id"]
    resultado["nivel_drift"] = metadatos["nivel_drift"]

    carpeta = RAIZ / "reports" / metadatos["batch_id"]
    carpeta.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(carpeta / "features.html"))
    snapshot.save_json(str(carpeta / "evidently.json"))

    (carpeta / "features.json").write_text(
        json.dumps(resultado, indent=2),
        encoding="utf-8",
    )

    if resultado["alerta"]:
        logger.warning(
            "ALERTA FEATURES: %s de %s variables presentan drift "
            "(%.1f%%); umbral: %.1f%%.",
            resultado["cantidad_con_drift"],
            resultado["total_features"],
            resultado["fraccion_con_drift"] * 100,
            resultado["umbral_fraccion"] * 100,
        )
    else:
        logger.info(
            "SIN ALERTA FEATURES: %s de %s variables presentan drift.",
            resultado["cantidad_con_drift"],
            resultado["total_features"],
        )

    print(json.dumps(resultado, indent=2))
    print(f"Reporte HTML: {carpeta / 'features.html'}")

    return resultado


if __name__ == "__main__":
    drift_features_flow()
    