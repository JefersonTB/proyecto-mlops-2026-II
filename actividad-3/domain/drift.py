import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift
from domain.modelo import validar_datos


def emular_drift(test, referencia, config):
    features = config["datos"]["features"]
    target = config["datos"]["target"]

    nivel = float(config["inferencia"]["nivel_drift"])
    tamano = int(config["inferencia"]["tamano_lote"])
    semilla = int(config["inferencia"]["random_state"])

    validar_datos(test, features, target)
    validar_datos(referencia, features, target)

    if not 0 <= nivel <= 1:
        raise ValueError("El nivel de drift debe estar entre 0 y 1")

    if tamano < 20:
        raise ValueError("El lote debe tener al menos 20 registros")

    # Cambiar la proporción de clase 1 mediante muestreo.
    tasa_referencia = float(referencia[target].mean())
    tasa_deseada = min(0.95, tasa_referencia + 0.40 * nivel)

    cantidad_1 = round(tamano * tasa_deseada)
    cantidad_0 = tamano - cantidad_1

    partes = []

    for clase, cantidad in [(0, cantidad_0), (1, cantidad_1)]:
        disponibles = test[test[target] == clase]

        if cantidad > len(disponibles):
            raise ValueError(
                f"No hay suficientes registros de clase {clase} "
                "para construir el lote sin repetir filas"
            )

        seleccion = disponibles.sample(
            n=cantidad,
            replace=False,
            random_state=semilla + clase,
        )
        partes.append(seleccion)

    lote = pd.concat(partes).sample(
        frac=1,
        random_state=semilla,
    ).copy()

    # Desplazar dos variables usando la escala del entrenamiento.
    for columna in ["variance", "entropy"]:
        desviacion = referencia[columna].std()
        lote[columna] += 3.0 * nivel * desviacion

    return lote.reset_index(drop=True)
def detectar_drift_target(referencia, lote, config):
    target = config["datos"]["target"]
    umbral = float(
        config["monitoreo"]["umbral_diferencia_target"]
    )

    if not 0 <= umbral <= 1:
        raise ValueError("El umbral del target debe estar entre 0 y 1")

    if target not in lote.columns:
        raise ValueError(
            f"El lote necesita etiquetas reales en '{target}' "
            "para evaluar drift del target"
        )

    # Aquí solo necesitamos validar la variable objetivo.
    validar_datos(referencia, [], target)
    validar_datos(lote, [], target)

    tasa_referencia = float(referencia[target].mean())
    tasa_lote = float(lote[target].mean())
    diferencia = abs(tasa_lote - tasa_referencia)

    return {
        "tasa_clase_1_referencia": tasa_referencia,
        "tasa_clase_1_lote": tasa_lote,
        "diferencia_absoluta": diferencia,
        "umbral": umbral,
        "alerta": diferencia > umbral,
        "filas_referencia": len(referencia),
        "filas_lote": len(lote),
    }
def detectar_drift_features(referencia, lote, config):
    features = config["datos"]["features"]
    p_umbral = float(config["monitoreo"]["p_valor_features"])
    umbral_fraccion = float(
        config["monitoreo"]["umbral_fraccion_features"]
    )

    validar_datos(referencia, features)
    validar_datos(lote, features)

    if not features or len(features) != len(set(features)):
        raise ValueError("La lista de features está vacía o repetida")

    if not 0 < p_umbral < 1:
        raise ValueError("El umbral del p-value debe estar entre 0 y 1")

    if not 0 <= umbral_fraccion <= 1:
        raise ValueError("El umbral de fracción debe estar entre 0 y 1")

    reporte = Report([
        ValueDrift(
            column=columna,
            method="ks",
            threshold=p_umbral,
        )
        for columna in features
    ])

    snapshot = reporte.run(
        current_data=lote[features],
        reference_data=referencia[features],
    )

    metricas = snapshot.dict()["metrics"]

    if len(metricas) != len(features):
        raise RuntimeError(
            "Evidently devolvió un número inesperado de métricas"
        )

    detalle = {}

    for columna, metrica in zip(features, metricas):
        p_valor = float(metrica["value"])

        detalle[columna] = {
            "p_valor": p_valor,
            "drift": p_valor < p_umbral,
        }

    cantidad = sum(
        resultado["drift"] for resultado in detalle.values()
    )
    fraccion = cantidad / len(features)

    resultado = {
        "metodo": "Kolmogorov-Smirnov",
        "p_umbral": p_umbral,
        "columnas": detalle,
        "cantidad_con_drift": cantidad,
        "total_features": len(features),
        "fraccion_con_drift": fraccion,
        "umbral_fraccion": umbral_fraccion,
        "alerta": fraccion > umbral_fraccion,
    }

    return resultado, snapshot