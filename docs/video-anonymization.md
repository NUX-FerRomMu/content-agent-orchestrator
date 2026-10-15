# Reemplazo facial con exportación verificada

El comando `anonymize-video` usa A ya registrado y B ya aprobado. No crea ni
cambia esas identidades. Produce un video nuevo; el original permanece intacto.
La voz se conserva y podrá tratarse mediante otro módulo.

Para el recorrido desde cero, incluyendo modelos y el asistente que arma el
JSON, consultar [quickstart-local.md](quickstart-local.md). `models prepare-swap`
descarga y verifica los dos ONNX de esta guía. `workflow create --interactive
--output clip.job.json` prepara el DAG y sus bindings; `workflow run clip.job.json`
lo ejecuta. `runs resume RUN_ID` recupera los snapshots/parametros originales
desde SQLite sin tener que volver a escribirlos.

## Preparación y ejecución

```bash
uv sync --locked --group swap
export PYTHONPATH=src
uv run --locked --group swap python -m content_agent_orchestrator --root /datos/influencer \
  anonymize-video /ruta/clip.mp4 \
  --swap-model /datos/influencer/artifacts/models/inswapper_128.onnx \
  --arcface-model /datos/influencer/artifacts/models/arcface_w600k_r50.onnx
```

CPU es el valor inicial. Para CUDA, instalar `--group swap-cuda` en lugar de
`--group swap` y ejecutar con `--device cuda`. Los grupos son excluyentes para
no instalar simultáneamente dos paquetes que proporcionan `onnxruntime`.
`--group generation` sigue siendo opcional e independiente para generar B.
El programa exige que el proveedor CUDA solicitado se inicialice; no cambia
silenciosamente a CPU. Esta etapa no establece rendimiento ni consumo de VRAM.

Los pesos se suministran explícitamente como archivos ONNX locales. El motor
es **InSwapper 128 FP32**, con ArcFace W600K R50 y su matriz de embeddings.
No sirven el checkpoint FP16 ni otro modelo que tenga firmas diferentes.
Cada archivo queda identificado por SHA-256 en las claves de caché y el informe;
se rechazan pesos externos, también dentro de tensores constantes/subgrafos.
La inferencia no descarga modelos, no usa servicios de imágenes y deshabilita
los eventos de telemetría de ONNX Runtime.

Los archivos probados en CPU provienen de
[FaceFusion assets, models-3.0.0](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.0.0):

| Archivo | Tamaño en bytes | SHA-256 verificado |
| --- | ---: | --- |
| `inswapper_128.onnx` | 555303150 | `a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7` |
| `arcface_w600k_r50.onnx` | 174388474 | `f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1` |

Descargarlos desde esa publicación, comprobar los hashes y guardarlos fuera del
checkout. No se distribuyen estos pesos con el código. ONNX Runtime y PyAV
están fijados en `uv.lock` como dependencias opcionales.

## Módulos y política de privacidad

El DAG completo tiene un único final, `export`:

```mermaid
flowchart LR
  inspect[video.inspect] --> detect[face.detect]
  detect --> track[face.track]
  track --> embed[face.embed]
  embed --> resolve[identity.resolve]
  resolve --> virtual[virtual.resolve]
  virtual --> swap[face.swap]
  swap --> validate[face.validate]
  validate --> redact[face.redact]
  redact --> export[video.export]
```

El [ejemplo declarativo](../examples/anonymize-video.json) incluye también las
dependencias convergentes de inspección, referencias y modelos. Puede usarse con
`run --input NAME=PATH` o extenderse con nuevos módulos antes del final.

- `face.swap` alinea el rostro mediante cinco landmarks, calcula la identidad
  virtual con ArcFace y aplica InSwapper. Sus píxeles son **intermedios sin
  validación**, incluso si el motor informó éxito.
  Declara `can_be_final=False`: el parser impide convertirlo accidentalmente
  en el nodo final de un DAG y publicar frames sin comprobar.
- `face.validate` vuelve a detectar rostros en los píxeles resultantes, con
  YuNet a mayor resolución y menor umbral de detección que la primera pasada.
  Exige correspondencia espacial única y cobertura de todos los rostros.
  Calcula SFace nuevo y exige similitud menor que **0.3** contra todas las
  referencias reales del snapshot y los rostros originales del frame,
  similitud de al menos **0.45** con B y margen B−A de al menos **0.15**.
- `face.redact` sustituye el frame completo por negro si falla el swap, la
  detección o la comprobación de identidad, si hay identidades desconocidas,
  falta B, hay ambigüedad o el detector original no encontró rostros.
  No elimina frames ni reutiliza píxeles originales como fallback público.
- `video.export` codifica solamente esos frames, normaliza físicamente la
  orientación y copia únicamente los streams de audio del original. Excluye
  otros videos, miniaturas, subtítulos, capítulos y metadatos del contenedor.
  Decodifica el archivo final y repite las comprobaciones de identidad en cada
  frame aceptado; verifica que los frames ocultados continúen opacos.

Un fallo de la comprobación del archivo codificado interrumpe la ejecución:
el archivo queda en temporales y no se publica. Si ningún frame pasa, tampoco
se publica por defecto. `--allow-all-redacted` permite producir explícitamente
un video totalmente negro, con el resultado indicado en el informe.

La carpeta `outputs/<run_id>/` contiene solamente video, informe y auditoría
de entrega. Los informes no contienen embeddings ni imágenes originales.
La publicación conjunta y su rollback usan el mecanismo atómico del Runner.

## Calidad, tiempos y audio

`--codec h264 --crf 18` produce MKV de buena calidad y tamaño moderado.
`--codec ffv1` conserva sin pérdida los píxeles RGB de 8 bits procesados, con
archivos considerablemente mayores. H.264 vuelve a comprimir el video;
conservar la resolución no equivale a conservar todos los detalles faciales.
InSwapper sintetiza un crop de **128×128**: la calidad de un primer plano
necesita evaluación visual y puede requerir otro motor/refinador más adelante.

Se preservan el número y orden de frames y sus tiempos de presentación,
incluyendo FPS variables. MKV cuantiza los tiempos a milisegundos; se exige
error de como máximo 1 ms y orden estrictamente creciente. Se conserva la
duración final publicada por el paquete fuente; si no existe, se infiere del
último intervalo positivo y se registra esa decisión en el informe.
Un clip de un único frame sin duración explícita se rechaza.
No se reduce la resolución: clips con rotación de 90°/270° intercambian ancho
y alto para conservar la resolución de visualización en orientación normal.

Se verifican codecs, tasas de muestreo, canales, conteos y hashes de payloads
de audio, además del timestamp de cada paquete. Un cambio o un audio que no
pueda copiarse/verificarse detiene la exportación. La voz original permanece.
La entrada soportada es **SDR de 8 bits**. HDR y mayor profundidad de bits se
rechazan; H.264 requiere dimensiones pares. FFV1 permite dimensiones impares.
La matriz/rango de color de la conversión RGB→H.264 se indican explícitamente.

## Caché y almacenamiento

Swap y ocultación almacenan bloques FFV1/NUT sin pérdida. El checkpoint sólo
acepta bloques completos cuyos hashes de video y registros coincidan.
Una interrupción permite reutilizarlos y recalcular el bloque incompleto;
el decoder recorre el original, pero el swap completado no vuelve a ejecutarse.
La RAM contiene un frame y los buffers de los códecs, sin cargar el clip entero.
`--chunk-frames` cambia el tamaño inicial de 32 frames; cada bloque se limita
a 4 GiB. Validación y exportación fallidas reinician su pasada, reutilizando
los módulos previos completados. `--timeout-seconds` ajusta los límites de los
subprocesos de inspección/decodificación/muxing.

Para reanudar, conservar `--snapshot` y `--catalog` originales y todos los
parámetros/modelos de la ejecución. `--cache fresh` elimina el trabajo fallido
de esa ejecución. Como en otros pipelines, los temporales se borran al publicar
correctamente cada módulo; los artifacts intermedios permanecen disponibles.
Actualizar el registro o aprobar otro B requiere una ejecución nueva con sus
snapshots actuales, sin cambiar retroactivamente ejecuciones históricas.

Los videos intermedios pueden contener el rostro original. Artifacts, temps,
embeddings, SQLite y backups necesitan almacenamiento privado. Los archivos
publicados tienen permisos 0600 y sus directorios 0700 en POSIX. Los archivos
sin pérdida y las copias necesarias para publicación pueden ocupar varias
veces el tamaño del original; prever espacio antes de ejecutar videos largos.

Estas comprobaciones son heurísticas de visión. Sus umbrales y calidad deben
evaluarse con los clips del operador; no prueban anonimato frente a cualquier
observador ni cubren voz, cuerpo o contexto. El informe permite distinguir
frames reemplazados y ocultados sin debilitar automáticamente los umbrales.
