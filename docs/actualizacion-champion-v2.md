
## Actualización final del champion

Se entrenó la regresión logística conjunta con C=0.01.
El AUC de validación aumentó de 0.701514 a 0.777021.
La regla de promoción registró la versión 2 y actualizó el alias champion.

Se inició una nueva instancia de la API en el puerto 8001,
porque el puerto 8000 seguía ocupado por la instancia anterior.
GET /health y POST /predict confirmaron la versión 2.
Para F-12, la predicción de ejemplo produjo score 0.2833736534
y clasificación 0 con umbral 0.5.

Evidencias: api_salud_v2.json y api_respuesta_v2.json,
en docs/evidencias/actividad-2/.

Los resultados de prueba publicados para la versión 1 corresponden
exclusivamente a esa versión. Esta promoción se basó en validación
y no demuestra una mejora independiente en datos futuros.
