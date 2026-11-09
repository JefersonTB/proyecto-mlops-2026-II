import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.datos_produccion import (
    ROOT, preparar_datos, seleccionar_variables
)

datos = preparar_datos().dropna(subset=["objetivo"]).copy()
datos["objetivo"] = datos["objetivo"].astype(int)
datos = datos.loc[datos["pozo"].isin(["15/9-F-12", "15/9-F-14"])]
columnas = seleccionar_variables(datos)

ventanas = [
    ("2012-06-26", "2013-06-26"),
    ("2013-06-26", "2014-06-26"),
    ("2014-06-26", "2015-06-23"),
]

def construir(nombre, columnas):
    numericas = [c for c in columnas if c != "pozo"]
    bloques = [
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(
                strategy="median", keep_empty_features=True
            )),
            ("escalar", StandardScaler()),
        ]), numericas),
    ]
    if "pozo" in columnas:
        bloques.append((
            "pozo",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["pozo"],
        ))

    if nombre.startswith("logistica"):
        modelo = LogisticRegression(
            C=float(nombre.split("_")[1]),
            class_weight="balanced",
            max_iter=3000,
            random_state=42,
        )
    else:
        modelo = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([
        ("preparar", ColumnTransformer(bloques)),
        ("modelo", modelo),
    ])


if __name__ == "__main__":
    filas = []
    nombres = ["logistica_0.01", "logistica_0.1", "logistica_1", "bosque"]
    pozos = ["15/9-F-12", "15/9-F-14"]

    for inicio, fin in ventanas:
        inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
        train = datos.loc[
            datos["fecha"] < inicio - pd.Timedelta(days=30)
        ]
        val = datos.loc[
            (datos["fecha"] >= inicio) & (datos["fecha"] < fin)
        ]

        for alcance in ["conjunto", "individual"]:
            for nombre in nombres:
                resultados = []

                grupos = [pozos] if alcance == "conjunto" else [[p] for p in pozos]
                for grupo in grupos:
                    tr = train.loc[train["pozo"].isin(grupo)]
                    va = val.loc[val["pozo"].isin(grupo)].copy()

                    if (
                        len(tr) < 365
                        or tr["objetivo"].value_counts().reindex(
                            [0, 1], fill_value=0
                        ).min() < 30
                        or va.empty
                    ):
                        continue

                    cols = [
                        c for c in columnas
                        if alcance == "conjunto" or c != "pozo"
                    ]
                    modelo = construir(nombre, cols)
                    modelo.fit(tr[cols], tr["objetivo"])
                    indice = list(modelo.classes_).index(1)
                    va["score"] = modelo.predict_proba(va[cols])[:, indice]
                    resultados.append(va)

                # Comparar solo cuando hay predicciones para ambos pozos.
                if not resultados:
                    continue
                pred = pd.concat(resultados)
                if set(pred["pozo"]) != set(pozos):
                    continue

                aucs = []
                for pozo in pozos:
                    g = pred.loc[pred["pozo"].eq(pozo)]
                    if g["objetivo"].nunique() == 2:
                        aucs.append(roc_auc_score(g["objetivo"], g["score"]))

                if len(aucs) != 2:
                    continue

                filas.append({
                    "inicio_validacion": str(inicio.date()),
                    "alcance": alcance,
                    "modelo": nombre,
                    "ejemplos": len(pred),
                    "auc_global": roc_auc_score(pred["objetivo"], pred["score"]),
                    "auc_F12": aucs[0],
                    "auc_F14": aucs[1],
                    "auc_promedio_pozos": sum(aucs) / 2,
                    "average_precision": average_precision_score(
                        pred["objetivo"], pred["score"]
                    ),
                })
                print("Completado:", inicio.date(), alcance, nombre)

    resultado = pd.DataFrame(filas)
    if resultado.empty:
        raise ValueError("No hubo ventanas con suficientes ejemplos.")

    carpeta = ROOT / "docs/evidencias/actividad-2"
    carpeta.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(
        carpeta / "comparacion_temporal.csv", index=False, encoding="utf-8"
    )

    resumen = resultado.groupby(["alcance", "modelo"]).agg(
        ventanas=("auc_global", "count"),
        auc_global_medio=("auc_global", "mean"),
        auc_global_minimo=("auc_global", "min"),
        auc_F12_medio=("auc_F12", "mean"),
        auc_F14_medio=("auc_F14", "mean"),
        auc_promedio_pozos=("auc_promedio_pozos", "mean"),
    ).reset_index()

    print("\nRESULTADOS POR VENTANA:")
    print(resultado.round(4).to_string(index=False))
    print("\nRESUMEN:")
    print(resumen.round(4).to_string(index=False))
    print("\nComparacion guardada. Champion sin modificaciones.")
