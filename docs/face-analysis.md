# Análisis de rostros

El pipeline detecta rostros y sigue su continuidad geométrica dentro de un clip.
La identificación de una persona entre videos y la anonimización pertenecen a
módulos posteriores. Un frame sin detecciones **no prueba que no haya un rostro**.

## Ejecución

Desde el checkout, con `PYTHONPATH=src` y `uv sync --locked` ejecutados:

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  analyze-faces /ruta/al/video.mp4
```

Las rutas de inputs relativas siguen refiriéndose al directorio de trabajo;
`--root` configura el almacenamiento. El detector predeterminado es el modelo
YuNet incluido en el paquete. `--model` puede apuntar a otra copia de **los mismos
pesos verificados**; otros modelos necesitan otra implementación/versionado.

Para ejecutar o adaptar el DAG explícito:

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  run examples/analyze-faces.json \
  --input video=/ruta/al/video.mp4 \
  --input detector=src/content_agent_orchestrator/assets/yunet/face_detection_yunet_2023mar.onnx
```

`face.report` es el único nodo final. Las etapas anteriores guardan resultados en
`artifacts/runs/`; todos los archivos del informe final van a `outputs/<run_id>/`.

## Módulos y contratos

| Módulo | Inputs | Outputs |
| --- | --- | --- |
| `face.detect` | Video, manifiesto de inspección, timestamps y modelo YuNet | Detecciones JSONL por frame y resumen JSON |
| `face.track` | Detecciones y su resumen | Frames con tracks, resumen de cada track, intervalos y resumen global |
| `face.report` | Los cuatro outputs de tracking | Informe JSON y los tres JSONL de detalle |

El manifiesto de inspección identifica el source por SHA-256 y contiene el hash
del archivo de timestamps. Los resúmenes de detección y tracking también contienen
los hashes de sus archivos asociados. Cada módulo comprueba esos vínculos antes
de consumirlos para rechazar combinaciones accidentales de artifacts distintos.

La detección usa OpenCV DNN en CPU, una imagen de trabajo de tamaño limitado y
los pesos oficiales YuNet 2023mar. Antes de cargar el modelo se verifica su
SHA-256. La versión de OpenCV participa en la versión del módulo y por lo tanto
en los fingerprints y requisitos de reanudación. La licencia y procedencia están
en `src/content_agent_orchestrator/assets/yunet/`.

FFmpeg decodifica por streaming y mantiene el orden de presentación mediante
`-fps_mode passthrough`. La rotación se aplica a la imagen de trabajo antes de
detectar; las cajas y landmarks se transforman de regreso a las coordenadas
codificadas originales. El procesamiento no recodifica, gira ni modifica el
archivo fuente. Rotaciones que no sean múltiplos de 90° se rechazan.

## Parámetros

En el campo `parameters` del nodo `face.detect`:

| Parámetro | Valor predeterminado | Significado |
| --- | --- | --- |
| `max_dimension` | 640 | Máxima dimensión de trabajo: múltiplo de 32 entre 64 y 2048. Las dimensiones efectivas se ajustan a múltiplos de 32. |
| `score_threshold` | 0.85 | Confianza mínima de YuNet, en `(0, 1]`. |
| `nms_threshold` | 0.3 | Umbral de supresión de detecciones superpuestas, en `(0, 1]`. |
| `top_k` | 100 | Máximo de candidatos de YuNet, entre 1 y 500. |
| `chunk_frames` | 64 | Frames por checkpoint, entre 1 y 1024. |
| `timeout_seconds` | 600 | Límite de la sesión de decoding/inferencia, entre 1 y 86400 segundos. |

La CLI permite ajustar los dos primeros con `--max-dimension` y
`--score-threshold`. La reducción afecta solamente a la imagen de trabajo del
detector, no a la resolución del video. Rostros pequeños pueden requerir una
dimensión mayor y más tiempo de procesamiento.

En el nodo `face.track`:

| Parámetro | Valor predeterminado | Significado |
| --- | --- | --- |
| `min_iou` | 0.3 | Superposición mínima para proponer una asociación, en `(0, 1]`. |
| `ambiguity_margin` | 0.05 | Diferencia máxima entre scores IoU que se considera ambigua, en `[0, 1]`. |
| `max_gap_frames` | 3 | Máximo de frames intermedios sin observación, entre 0 y 300. |
| `max_gap_seconds` | 0.5 | Máximo de tiempo entre observaciones al recuperar una trayectoria, entre 0 y 30 segundos. |

Se usa el timestamp entero y el time base para calcular tiempos, incluyendo FPS
variables. La asociación exige una mejor correspondencia mutua por IoU. Cuando
varias alternativas son demasiado similares se cierran las trayectorias
afectadas y se crean IDs nuevos marcados como ambiguos. Una recuperación tras
un hueco queda marcada como `reacquired`, con `gap_frames` explícito.

## Archivos finales

- `report.json`: parámetros, source hash, geometría, modelo, counts y referencias
  a los JSONL. Declara `anonymization_performed: false` e
  `identity_recognition_performed: false`.
- `frames.jsonl`: un registro por frame con `frame_index`,
  `presentation_timestamp`, `presentation_time` y `faces`. Cada rostro contiene
  `bbox: [x, y, width, height]`, cinco `landmarks`, `score`, `track_id`,
  `association` y `gap_frames`.
- `tracks.jsonl`: primera/última observación, cantidad de observaciones, huecos
  y razón de cierre (`end_of_clip`, `gap_expired` o `ambiguous`) de cada track.
- `intervals.jsonl`: intervalos inclusivos de frames con `no_faces` o
  `ambiguous_association`, junto con sus timestamps observados.

Las coordenadas están expresadas en píxeles **antes de la rotación de display**,
identificados como `coded_pixels`; el informe conserva `rotation_degrees` para
que un consumidor transforme las anotaciones al mostrarlas. No se guardan
imágenes de rostros ni embeddings. Los resúmenes/intervalos se escriben de forma
incremental; no se conserva una lista de imágenes o trayectorias completas en RAM.

## Recuperación y errores

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  analyze-faces /ruta/al/video.mp4 --resume <run_id> --cache reuse
```

La detección confirma cada bloque con un archivo JSONL y un checkpoint atómico
que incluye su tamaño, hash y cantidad de frames. Conserva el prefijo íntegro;
un bloque ausente o corrupto y los posteriores se vuelven a procesar. Los
archivos `.part` nunca cuentan como bloques terminados. El fingerprint incluye
los inputs, parámetros y versión del detector.

Al reanudar se vuelve a decodificar desde el comienzo y se omite la inferencia
de los bloques confirmados. Esto ahorra inferencia sin asumir que un seek de un
codec interframe conserva exactamente la numeración y los timestamps. El
tracking y el informe repiten su pasada de metadatos si fallan. `--cache fresh`
descarta el trabajo temporal de esa ejecución según las reglas del coordinador.

Se rechazan un modelo modificado, timestamps ausentes o no crecientes, registros
inválidos, errores de decoding y diferencias entre la cantidad de timestamps y
frames reales. Los procesos FFmpeg se terminan ante errores o timeout. El
coordinador mantiene los temporales ante fallos y publica resultados solamente
cuando el nodo final termina correctamente.

El tracking geométrico puede equivocarse, especialmente durante cruces u
oclusiones; sus IDs no deben convertirse directamente en identidades de personas.
La siguiente etapa agregará referencias biométricas y resolución de identidades.
