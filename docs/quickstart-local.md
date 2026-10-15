# Del video original al video con tu identidad virtual

El flujo completo ya produce un video nuevo con una B estable, auditoría por
frame y un informe de privacidad. Usa modelos locales: YuNet + SFace para A,
RealVisXL/SDXL para generar B una vez e InSwapper128 + ArcFace para el reemplazo.
Cada video se ejecuta como un DAG con **un solo nodo final**, `video.export`.
No necesita una API de OpenAI ni envía imágenes a un servicio de generación.

Hay dos revisiones visuales: elegir el track que corresponde a A y aprobar el
retrato sintético B. El programa guarda esa asociación; los siguientes videos
reutilizan exactamente el mismo rostro. El asistente arma el JSON del pipeline
y sus entradas, sin que tengas que escribirlo a mano.

## 1. Instalar y comprobar el entorno

Requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/) y FFmpeg con FFprobe en
`PATH`. Por ejemplo, en Ubuntu instalar `ffmpeg`; en Windows instalar una
distribución de FFmpeg y agregar su directorio `bin` al `PATH`.

Desde el checkout, para tu NVIDIA y generación de B:

```bash
uv sync --locked --group generation --group swap-cuda
source .venv/bin/activate
export CONTENT_AGENT_HOME="$HOME/influencer-data"
content-agent init
content-agent doctor --device cuda --generation
```

PowerShell usa estos equivalentes para activar el entorno y elegir la carpeta:

```powershell
.venv\Scripts\Activate.ps1
$env:CONTENT_AGENT_HOME = "$HOME\influencer-data"
content-agent init
content-agent doctor --device cuda --generation
```

Los comandos posteriores `content-agent ...` son iguales en ambos sistemas.
Los ejemplos con `\` para continuar líneas son de Bash; en PowerShell escribir
la orden en una línea o usar el acento grave como continuación.

Para CPU, instalar `--group swap` en lugar de `--group swap-cuda`, y usar
`--device cpu` al generar B y construir el job:

```bash
uv sync --locked --group generation --group swap
source .venv/bin/activate
content-agent doctor --device cpu --generation
```

CPU permite probar el flujo, pero SDXL puede ser muy lento y necesita bastante
RAM. El diseño de generación usa offload y apunta a 16 GB de VRAM; no hay una
medición de GPU en este entorno. El nombre exacto de tu placa y sus drivers
determinarán si el stack CUDA es compatible. `doctor` comprueba disponibilidad;
el motor además exige que las sesiones ONNX se inicialicen con CUDA cuando se
solicita. Un fallo de CUDA se informa; elegir CPU requiere hacerlo explícito.
Si faltan bibliotecas CUDA/cuDNN, instalar las versiones que pide la versión de
ONNX Runtime GPU del lockfile según su documentación de instalación.

El paquete se instala editable y provee `content-agent`; **ya no hace falta
`PYTHONPATH`**. No ejecutar `uv run` sin los grupos si querés conservar el stack
opcional instalado: `uv` sincroniza el entorno y puede quitar esos paquetes.
Alternativa a activar `.venv`:

```bash
uv run --locked --group generation --group swap-cuda content-agent --help
```

Todos los comandos aceptan `--root /ruta/privada` **antes** del subcomando para
reemplazar `CONTENT_AGENT_HOME`. Si no se configura ninguna de las dos opciones,
se usa `~/.content-agent-orchestrator`.

## 2. Preparar los modelos una vez

```bash
content-agent models prepare-swap
content-agent virtual prepare-model --model realvis-xl
```

El primer comando descarga aproximadamente **696 MiB** desde la publicación
`facefusion-assets/models-3.0.0`, verifica tamaños y SHA-256 fijados en el código
y publica ambos ONNX juntos en
`artifacts/models/inswapper-128-models-3.0.0/`. YuNet y SFace ya vienen incluidos.
Repetirlo con la copia íntegra funciona sin consultar la red. Si falla una
descarga, el siguiente intento reutiliza archivos completos verificados y
reinicia solamente los archivos incompletos. Un reemplazo fallido conserva la
copia anterior. Los pesos no se agregan al repositorio.

El segundo comando prepara **RealVisXL V5.0**, un checkpoint fotorealista SDXL
de aproximadamente **6.5 GiB**. Devuelve `model_dir`: guardar esa ruta para el
paso 5. La revisión upstream se resuelve a un commit y cada archivo se verifica.
También admite `--model sdxl-base` o `--revision COMMIT`. Puede requerir acceso
a `huggingface.co` y a los hosts de sus redirecciones, por ejemplo
`us.aws.cdn.hf.co`. Un error HTTP o de proxy deja los archivos completos para
reintentar el mismo comando; no publica un modelo incompleto.

Las descargas son explícitas. Ejecutar el pipeline o crear una imagen desde
un modelo ya preparado no vuelve a descargar pesos. Para verificar el swap:

```bash
content-agent doctor --device cuda --models --generation
```

También se pueden suministrar ONNX existentes con `workflow create
--swap-model /ruta/inswapper_128.onnx --arcface-model /ruta/arcface_w600k_r50.onnx`.
Deben ser el modelo FP32 original y la firma compatible, sin pesos externos.
Los hashes de esos archivos quedan congelados en el job y en la caché.

## 3. Importar y analizar un clip de referencia

Para empezar conviene un clip corto donde estés solo, de frente, bien iluminado
y con el rostro suficientemente grande. No tenés que conocer resolución ni FPS:
la inspección obtiene esos datos y recorre los tiempos reales, incluidos VFR.

```bash
content-agent artifacts import /ruta/a/referencia.mp4
```

La respuesta devuelve `path`: una copia privada en `artifacts/raw_footages/`
cuyo nombre incluye el hash. El archivo original permanece intacto. Repetir la
importación de ese mismo archivo no duplica ni pisa la copia. También podés usar
directamente una ruta externa; la importación es una comodidad.

Usar esa ruta, o la del archivo original:

```bash
content-agent recognize-faces /ruta/a/referencia.mp4
```

El JSON final devuelve `run_id`. El progreso está en stderr; stdout contiene
JSON, de modo que se puede guardar con `> recognition.json`. Para encontrar
ejecuciones e identificar el track a registrar:

```bash
content-agent runs list
content-agent identity tracks --run RUN_ID_RECONOCIMIENTO
content-agent runs show RUN_ID_RECONOCIMIENTO
```

`identity tracks` devuelve los IDs y el número de observaciones utilizables.
Elegir un track que corresponda sólo a tu rostro. El ejemplo habitual de un
clip con una sola persona es `track-000001`, pero no hay que asumirlo cuando
aparecen otras personas. Los outputs de reconocimiento incluyen el detalle
por frame/track para contrastarlo con el video. Se necesitan al menos tres
observaciones de calidad y coherentes; el alta rechaza mezclas ambiguas.

## 4. Registrar A

```bash
content-agent identity register --from-run RUN_ID_RECONOCIMIENTO \
  --track track-000001 --alias mi-identidad
content-agent identity list
```

Guardar el `identity_id` devuelto. Las referencias biométricas viven en SQLite
bajo `state/`, fuera de los artifacts del DAG. Los módulos consumen snapshots
inmutables para que sus resultados sean reproducibles y cacheables.

Si otro clip de referencia mejora la cobertura de tu rostro, reconocerlo y
añadir un track coherente al mismo ID:

```bash
content-agent identity register --from-run OTRO_RUN_ID \
  --track track-000001 --identity-id ID_A
```

No hace falta registrar de nuevo A para cada video. Si cambia el registro,
crear un job nuevo para los videos siguientes; los jobs anteriores conservan
sus snapshots históricos. `--from-run` verifica los artifacts guardados antes
de usarlos. Se mantiene la alternativa `--embeddings ARCHIVO --summary ARCHIVO`.

## 5. Definir y generar B

El perfil se puede completar con preguntas:

```bash
content-agent profile create --interactive --output perfil.json
```

O con una orden reproducible, ajustando género y tono para que correspondan a A:

```bash
content-agent profile create --output perfil.json \
  --gender woman --skin-tone 6 --undertone warm --age-years 35 \
  --features "natural makeup, subtle eyeliner"
```

`gender` acepta `woman`, `man`, `androgynous`; `skin-tone` es la escala Monk
1–10, de claro a oscuro. La edad virtual permitida es 25–65. Los atributos se
declaran explícitamente; no se infieren del rostro. Preferencias cortas en
inglés suelen funcionar mejor con este checkpoint. El programa rechaza un
prompt que desborde los tokenizadores para evitar truncar requisitos.

Con `ID_A` y el `model_dir` devuelto en el paso 2:

```bash
content-agent virtual create ID_A --profile perfil.json \
  --model-dir /ruta/devuelta/por/prepare-model --device cuda
```

Usa 1024×1024, 30 pasos, semilla 42, CPU offload y VAE slicing/tiling. La respuesta
devuelve `portrait_path` y `candidate_id`. Abrir el PNG y revisar que sea una
cara ficticia adulta, sin anteojos ni rasgos excéntricos, y que el género, tono
y cosmética correspondan al perfil. Ningún comando la aprueba automáticamente:

```bash
content-agent virtual approve ID_A CANDIDATE_ID --reviewed
content-agent virtual list
```

La aprobación guarda los píxeles exactos en la DB. Repetir `create` con el mismo
perfil reutiliza B; no depende de regenerarla con una semilla. Una asociación
aprobada no se sobrescribe silenciosamente. Una generación no demuestra que
nadie real en el mundo se parezca a B; se exige procedencia ficticia, revisión
visual y separación biométrica de todas las personas del registro local.

Si un candidato no sirve, todavía **sin aprobar**, usar otra `--seed` y repetir
la generación; genera otro pedido. Si falló la generación, repetir el mismo
comando usa su caché; `--cache fresh` descarta los temporales de ese pedido
fallido. No pisa candidaturas publicadas ni asociaciones aprobadas.

## 6. Armar y ejecutar el pipeline

Después de registrar A y aprobar B, el asistente solicita ruta del video,
dispositivo y codec, y escribe el job completo:

```bash
content-agent workflow create --interactive --output clip.job.json
content-agent workflow run clip.job.json
```

Para elegir CUDA sin responder a todas las preguntas o automatizar el armado:

```bash
content-agent workflow create --output clip.job.json \
  --video /ruta/a/clip.mp4 --device cuda --codec h264 \
  --crf 18 --chunk-frames 32 --timeout-seconds 600
content-agent workflow run clip.job.json
```

El job contiene `pipeline` (DAG), `inputs` (rutas absolutas, tamaños, SHA-256),
`storage_root` y versión del formato. Congela los snapshots de A y B actuales y
usa los modelos preparados del paso 2. No sobrescribe un job existente. Los
inputs deben existir en almacenamiento permanente, nunca en `temps/` o `state/`.
Se puede ejecutar desde otro directorio: sus bindings son absolutos. Se exige
el mismo `CONTENT_AGENT_HOME` o `--root` usado al crearlo. Si cambia el contenido
de una entrada, se rechaza antes de ejecutar; restaurarla o crear otro job.

La respuesta final devuelve las rutas de tres outputs: video MKV, informe JSON
y auditoría JSONL por frame, bajo `outputs/<run_id>/`. Consultar `counts` en
el informe: `validated` son frames reemplazados y verificados; `redacted` son
frames ocultados. El archivo final se vuelve a decodificar y comprobar antes
de publicarse. Una corrida que no valida ningún frame falla por defecto;
`--allow-all-redacted` al crear el job permite expresamente un video todo negro.

El procesamiento usa bloques para mantener acotada la memoria. Cada frame que
falla se reemplaza **por negro completo**, conservando número de frames y tiempos.
Es más conservador que un blur facial y no altera los FPS. Un fallo posterior
en la validación del archivo codificado impide toda publicación de esa entrega.

## 7. Reutilizar y recuperar

```bash
# Nueva ejecución: reutiliza nodos completos cuyos inputs y versiones coincidan.
content-agent workflow run clip.job.json

# Ver estado, error y rutas de outputs/intermedios de una ejecución.
content-agent runs list --limit 20
content-agent runs show RUN_ID

# Recuperar un fallo/interrupción usando DAG y bindings originales de SQLite.
content-agent runs resume RUN_ID

# Reiniciar ese fallo sin reutilizar sus temporales ni nodos completos.
content-agent runs resume RUN_ID --cache fresh

# Nueva ejecución sin usar caché previa.
content-agent workflow run clip.job.json --cache fresh
```

La reanudación no necesita el job ni repetir `--snapshot`/`--catalog`: recupera
las rutas registradas por el Runner. Esos archivos sí tienen que seguir
existiendo. Requiere versiones de módulos y configuración idénticas; una
ejecución exitosa se repite como una nueva corrida, no se reanuda. Si otro
proceso está usando el mismo ID, el lock impide reanudarlo simultáneamente.

Swap reutiliza bloques completos con checksums verificados; un bloque a medio
escribir se recalcula. La validación/exportación fallidas repiten su pasada,
reutilizando los módulos anteriores completos. Si cambió un input de una
ejecución fallida, el Runner invalida la caché afectada y descendiente; para
ejecutar un job guardado sigue rigiendo además su verificación de hashes.

Para el siguiente video, usar la misma raíz, registrar sólo si es necesario y
crear otro job con `--video NUEVO_CLIP`. B ya aprobada se reutiliza. Los artifacts
generados también pueden conectarse a otros DAGs como inputs; no hace falta
copiarlos a `outputs` para usarlos.

## Pipelines propios y ayuda

```bash
content-agent --help
content-agent workflow create --help
content-agent virtual create --help
content-agent modules
content-agent run examples/inspect-video.json --input video=/ruta/a/clip.mp4
```

`modules` enumera contratos de inputs/outputs y si el módulo admite ser final.
Los [ejemplos JSON](../examples/) son DAGs puros y se ejecutan con `run` y
bindings `--input NOMBRE=RUTA`; `workflow run` espera el job con bindings guardados.
También podés editar `pipeline` dentro de un job para conectar módulos propios,
manteniendo sus entradas y un solo final. Se valida el DAG antes de ejecutarlo.
El orden del JSON no determina el orden de ejecución; las dependencias sí.

Las rutas relativas en comandos y `run --input` se interpretan respecto del
directorio actual. `--root` no las remapea al almacenamiento. Los jobs creados
por el asistente evitan esa ambigüedad guardando rutas absolutas.

## Espacio, calidad y resolución de problemas

- **Espacio:** los intermedios FFV1 y copias de publicación pueden ocupar varias
  veces el original. Una corrida exitosa borra sus temporales, pero conserva
  artifacts intermedios para inspección y caché. No borrar archivos mientras
  haya una ejecución activa; no hay una purga automática que destruya corridas
  o snapshots históricos. `--cache fresh` no elimina artifacts anteriores.
- **Muchos frames negros:** consultar los motivos en la auditoría. Mejorar
  iluminación, tamaño del rostro y cobertura de referencias de A; verificar
  que B esté aprobada. Un rostro desconocido o sin target también oculta el
  frame entero. No se reducen automáticamente los umbrales para obtener salida.
- **Calidad:** no se reduce la resolución. H.264 CRF 18 vuelve a comprimir;
  `--codec ffv1` conserva los píxeles procesados sin pérdida y ocupa más espacio.
  InSwapper128 genera un crop de 128×128; un primer plano requiere evaluar
  visualmente detalle y estabilidad. No se promete calidad facial idéntica al
  original ni anonimato frente a cualquier observador.
- **Formatos:** se admiten fuentes SDR de 8 bits; HDR/profundidades mayores
  se rechazan. H.264 requiere dimensiones pares; FFV1 permite impares. Se
  preservan VFR y orientación de visualización, con tolerancia de timing de 1 ms.
- **Audio:** se copia y verifica sin intercambio de voz. Un codec que no se
  pueda copiar/verificar al MKV hace fallar la exportación. Voz, cuerpo y contexto
  pueden identificar al operador; el módulo de voz queda para otro feature.
- **Privacidad local:** originales, caché, embeddings, asociaciones A→B, DB y
  snapshots son privados. En POSIX se usan archivos 0600 y directorios 0700.
  Elegir una raíz fuera del repo y proteger también backups y permisos en Windows.
  Los outputs de análisis también son privados; para compartir, usar sólo el
  video final ya revisado de la corrida de anonimización.

Detalles de umbrales y contratos: [video-anonymization.md](video-anonymization.md),
[identity-recognition.md](identity-recognition.md) y
[virtual-identities.md](virtual-identities.md).
