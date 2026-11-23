import json
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, get_run_logger

from domain.drift import detectar_drift_target

RAIZ = Path(__file__).resolve().parents[2]


@flow(name="banknote-drift-target", log_prints=True)
def drift_evento_flow():
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

    resultado = detectar_drift_target(
        referencia,
        lote,
        config,
    )

    resultado["batch_id"] = metadatos["batch_id"]
    resultado["nivel_drift"] = metadatos["nivel_drift"]

    carpeta = RAIZ / "reports" / metadatos["batch_id"]
    carpeta.mkdir(parents=True, exist_ok=True)

    destino = carpeta / "target.json"
    destino.write_text(
        json.dumps(resultado, indent=2),
        encoding="utf-8",
    )

    if resultado["alerta"]:
        logger.warning(
            "ALERTA TARGET: cambio de %.2f puntos porcentuales; "
            "umbral de %.2f puntos.",
            resultado["diferencia_absoluta"] * 100,
            resultado["umbral"] * 100,
        )
    else:
        logger.info(
            "SIN ALERTA TARGET: cambio de %.2f puntos porcentuales.",
            resultado["diferencia_absoluta"] * 100,
        )

    print(json.dumps(resultado, indent=2))
    print(f"Reporte guardado en: {destino}")

    return resultado


if __name__ == "__main__":
    drift_evento_flow()