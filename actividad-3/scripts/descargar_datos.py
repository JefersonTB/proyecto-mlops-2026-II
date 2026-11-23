from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
COLUMNAS = ["variance", "skewness", "curtosis", "entropy", "class"]

# Resolver la ubicación del proyecto desde este archivo.
RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "raw" / "banknote.csv"


def main():
    with urlopen(URL, timeout=60) as respuesta:
        contenido = respuesta.read()

    with ZipFile(BytesIO(contenido)) as archivo:
        with archivo.open("data_banknote_authentication.txt") as datos:
            df = pd.read_csv(datos, header=None, names=COLUMNAS)

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DESTINO, index=False)

    print(f"Dataset guardado en: {DESTINO}")
    print(f"Filas y columnas: {df.shape}")
    print(f"Valores nulos: {df.isna().sum().sum()}")
    print(f"Filas duplicadas: {df.duplicated().sum()}")
    print("\nRegistros por clase:")
    print(df["class"].value_counts().sort_index())


if __name__ == "__main__":
    main()