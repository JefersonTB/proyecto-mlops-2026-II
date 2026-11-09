from prefect import serve
from prefect.schedules import Cron

from flows.entrenamiento_logistica import entrenamiento_logistica
from flows.inferencia_produccion import inferencia_produccion
from src.datos_produccion import leer_yaml


if __name__ == "__main__":
    cfg = leer_yaml("despliegues.yaml")

    entrenamiento = entrenamiento_logistica.to_deployment(
        name=cfg["entrenamiento"],
        description="Entrenamiento manual del modelo de caida a 30 dias.",
    )

    inferencia = inferencia_produccion.to_deployment(
        name=cfg["inferencia"],
        schedule=Cron(
            cfg["cron"],
            timezone=cfg["zona_horaria"],
        ),
        description=(
            "Inferencia diaria. Simulacion historica sobre la fecha "
            "de corte definida en configs/inferencia.yaml."
        ),
    )

    serve(entrenamiento, inferencia)
