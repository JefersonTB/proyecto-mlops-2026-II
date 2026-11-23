from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "raw" / "banknote.csv"
SALIDA = RAIZ / "data" / "processed"


def main():
    df = pd.read_csv(ENTRADA)
    originales = len(df)

    # Eliminar duplicados conservando el archivo original.
    df = df.drop_duplicates().copy()

    # Identificador para rastrear el origen de cada registro.
    df.insert(0, "source_id", df.index)

    train, test = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["class"],
    )

    SALIDA.mkdir(parents=True, exist_ok=True)
    train.to_csv(SALIDA / "train.csv", index=False)
    test.to_csv(SALIDA / "test.csv", index=False)

    print(f"Duplicados eliminados: {originales - len(df)}")
    print(f"Registros únicos: {len(df)}")
    print(f"Entrenamiento: {len(train)}")
    print(f"Prueba: {len(test)}")
    print(
        "IDs compartidos:",
        len(set(train["source_id"]) & set(test["source_id"])),
    )

    print("\nProporciones de clase en entrenamiento:")
    print(train["class"].value_counts(normalize=True).sort_index())

    print("\nProporciones de clase en prueba:")
    print(test["class"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()