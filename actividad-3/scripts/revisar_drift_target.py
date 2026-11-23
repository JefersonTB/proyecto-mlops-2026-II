import json
from pathlib import Path

import pandas as pd
import yaml

from domain.drift import detectar_drift_target

RAIZ = Path(__file__).resolve().parents[1]


def main():
    with (RAIZ / "configs" / "config.yaml").open(
        encoding="utf-8"
    ) as archivo:
        config = yaml.safe_load(archivo)

    referencia = pd.read_csv(
        RAIZ / config["datos"]["entrenamiento"]
    )

    resumen = []

    for ruta in sorted((RAIZ / "data" / "batches").glob("*.json")):
        metadatos = json.loads(ruta.read_text(encoding="utf-8"))
        lote = pd.read_csv(RAIZ / metadatos["ruta_lote"])

        resultado = detectar_drift_target(
            referencia, lote, config
        )
        resultado["batch_id"] = metadatos["batch_id"]
        resultado["nivel_drift"] = metadatos["nivel_drift"]

        carpeta = RAIZ / "reports" / metadatos["batch_id"]
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "target.json").write_text(
            json.dumps(resultado, indent=2),
            encoding="utf-8",
        )

        resumen.append({
            "batch_id": metadatos["batch_id"],
            "nivel": metadatos["nivel_drift"],
            "tasa_clase_1": resultado["tasa_clase_1_lote"],
            "cambio_puntos_porcentuales": (
                resultado["diferencia_absoluta"] * 100
            ),
            "alerta": resultado["alerta"],
        })

    if not resumen:
        raise ValueError("No se encontraron lotes guardados")

    tabla = pd.DataFrame(resumen).sort_values("nivel")
    tabla.to_csv(
        RAIZ / "reports" / "resumen_target.csv",
        index=False,
    )
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()