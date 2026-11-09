from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def leer_yaml(nombre):
    with open(ROOT / "configs" / nombre, encoding="utf-8-sig") as archivo:
        return yaml.safe_load(archivo)


def preparar_datos():
    cfg = leer_yaml("produccion.yaml")
    df = pd.read_excel(
        ROOT / cfg["datos"]["archivo"],
        sheet_name=cfg["datos"]["hoja"],
    )
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"], errors="raise")

    if df.duplicated(["NPD_WELL_BORE_NAME", "DATEPRD"]).any():
        raise ValueError("Hay registros duplicados por pozo y fecha.")

    df = df.loc[df["WELL_TYPE"].eq(cfg["datos"]["tipo_pozo"])].copy()
    historial = cfg["objetivo"]["historial_dias"]
    horizonte = cfg["objetivo"]["horizonte_dias"]
    umbral = cfg["objetivo"]["caida_relativa"]
    tablas = []

    sensores = {
        "ON_STREAM_HRS": "horas",
        "AVG_DOWNHOLE_PRESSURE": "presion_fondo",
        "AVG_DOWNHOLE_TEMPERATURE": "temperatura_fondo",
        "AVG_ANNULUS_PRESS": "presion_anular",
        "AVG_CHOKE_SIZE_P": "apertura_choke",
        "AVG_WHP_P": "presion_cabeza",
        "AVG_WHT_P": "temperatura_cabeza",
    }

    for pozo, grupo in df.groupby("NPD_WELL_BORE_NAME"):
        diario = grupo.set_index("DATEPRD").sort_index().asfreq("D")
        petroleo = pd.to_numeric(diario["BORE_OIL_VOL"], errors="raise")

        if petroleo.dropna().lt(0).any():
            raise ValueError(f"Produccion negativa en {pozo}.")

        pasado = petroleo.rolling(historial, min_periods=historial).mean()
        futuro = (
            petroleo.rolling(horizonte, min_periods=horizonte)
            .mean().shift(-horizonte)
        )

        tabla = pd.DataFrame(index=diario.index)
        tabla["pozo"] = pozo
        tabla["petroleo_media_30d"] = pasado
        tabla["petroleo_media_7d"] = petroleo.rolling(7, min_periods=7).mean()
        tabla["petroleo_std_30d"] = petroleo.rolling(
            historial, min_periods=historial
        ).std()
        tabla["tendencia_7d_30d"] = (
            tabla["petroleo_media_7d"] / pasado.where(pasado.gt(0))
        )
        tabla["fraccion_ceros_30d"] = (
            petroleo.eq(0).astype(float)
            .where(petroleo.notna())
            .rolling(historial, min_periods=historial).mean()
        )

        for columna, nombre in sensores.items():
            valores = pd.to_numeric(diario[columna], errors="raise")
            tabla[f"{nombre}_media_30d"] = valores.rolling(
                historial, min_periods=1
            ).mean()
            tabla[f"{nombre}_faltantes_30d"] = (
                valores.isna().astype(float)
                .rolling(historial, min_periods=historial).mean()
            )

        # Variables historicas adicionales; ninguna usa datos futuros.
        mes_previo = pasado.shift(historial)
        tabla["ratio_produccion_mes_anterior"] = (
            pasado / mes_previo.where(mes_previo.gt(0))
        )
        tabla["variabilidad_relativa"] = (
            tabla["petroleo_std_30d"] / pasado.where(pasado.gt(0))
        )

        agua = pd.to_numeric(diario["BORE_WAT_VOL"], errors="raise")
        liquido = petroleo + agua
        agua_30 = agua.rolling(historial, min_periods=historial).sum()
        liquido_30 = liquido.rolling(historial, min_periods=historial).sum()
        corte_agua = agua_30 / liquido_30.where(liquido_30.gt(0))

        tabla["corte_agua_30d"] = corte_agua
        tabla["cambio_corte_agua_30d"] = (
            corte_agua - corte_agua.shift(historial)
        )

        horas = pd.to_numeric(diario["ON_STREAM_HRS"], errors="raise")
        horas_7 = horas.rolling(7, min_periods=7).mean()
        horas_30 = horas.rolling(historial, min_periods=historial).mean()
        tabla["cambio_horas_7d_30d"] = horas_7 - horas_30

        presion = pd.to_numeric(diario["AVG_WHP_P"], errors="raise")
        presion_7 = presion.rolling(7, min_periods=1).mean()
        presion_30 = presion.rolling(historial, min_periods=1).mean()
        tabla["ratio_presion_7d_30d"] = (
            presion_7 / presion_30.where(presion_30.gt(0))
        )

        apertura = pd.to_numeric(diario["AVG_CHOKE_SIZE_P"], errors="raise")
        tabla["cambio_choke_7d_30d"] = (
            apertura.rolling(7, min_periods=1).mean()
            - apertura.rolling(historial, min_periods=1).mean()
        )

        # Los dias sin futuro conocido no tienen etiqueta.
        tabla["objetivo"] = pd.Series(pd.NA, index=tabla.index, dtype="Int64")
        etiquetables = pasado.gt(0) & futuro.notna()
        tabla.loc[etiquetables, "objetivo"] = (
            futuro.loc[etiquetables]
            .lt(pasado.loc[etiquetables] * (1 - umbral))
            .astype(int)
        )

        # Conservamos filas con historial completo y base positiva.
        # Algunas no tienen etiqueta: pueden servir para inferencia.
        tabla = tabla.loc[pasado.gt(0)]
        tabla.index.name = "fecha"
        tablas.append(tabla.reset_index())

    return pd.concat(tablas, ignore_index=True)


def seleccionar_variables(datos):
    cfg = leer_yaml("features.yaml")
    disponibles = [
        columna for columna in datos.columns
        if columna not in {"fecha", "objetivo"}
    ]
    alias = cfg["seleccion"].upper()

    if alias == "ALL":
        columnas = disponibles
    elif alias == "NUMERIC":
        columnas = [
            columna for columna in disponibles
            if pd.api.types.is_numeric_dtype(datos[columna])
        ]
    elif alias == "CUSTOM":
        columnas = cfg["custom"]
        if not columnas or not set(columnas).issubset(disponibles):
            raise ValueError("CUSTOM contiene variables invalidas o esta vacio.")
    else:
        raise ValueError("Seleccion debe ser ALL, NUMERIC o CUSTOM.")

    return columnas


def dividir_datos(datos):
    cfg = leer_yaml("produccion.yaml")
    etiquetados = datos.dropna(subset=["objetivo"]).copy()
    etiquetados["objetivo"] = etiquetados["objetivo"].astype(int)

    inicio_val = pd.Timestamp(cfg["fechas"]["inicio_validacion"])
    inicio_test = pd.Timestamp(cfg["fechas"]["inicio_prueba"])
    margen = pd.Timedelta(days=cfg["particion"]["separacion_dias"])

    if margen.days < cfg["objetivo"]["horizonte_dias"]:
        raise ValueError("La separacion debe cubrir el horizonte del pronostico.")

    fecha = etiquetados["fecha"]
    partes = {
        "entrenamiento": etiquetados.loc[fecha < inicio_val - margen],
        "validacion": etiquetados.loc[
            (fecha >= inicio_val) & (fecha < inicio_test - margen)
        ],
        "prueba": etiquetados.loc[fecha >= inicio_test],
    }

    for nombre, parte in partes.items():
        if parte["objetivo"].nunique() != 2:
            raise ValueError(f"{nombre} no contiene ambas clases.")
    return partes


if __name__ == "__main__":
    datos = preparar_datos()
    variables = seleccionar_variables(datos)
    partes = dividir_datos(datos)

    print("Variables seleccionadas:", variables)
    print("Numero de variables:", len(variables))
    for nombre, parte in partes.items():
        print(
            nombre,
            "ejemplos:", len(parte),
            "con caida:", int(parte["objetivo"].sum()),
        )
    print("OK: preparacion de datos completada")
