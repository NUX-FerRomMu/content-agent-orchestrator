# Content Agent Orchestrator

Coordinador de pipelines de módulos componibles. Cada pipeline es un DAG que
converge en **un único nodo final**. Permite inspeccionar videos, detectar y
seguir rostros dentro de cada clip, reconocer personas registradas entre videos
y generar identidades virtuales persistentes. El intercambio facial y la
validación de anonimización del video serán etapas posteriores.

## Requisitos y preparación

- Python 3.12, según `.python-version`.
- [uv](https://docs.astral.sh/uv/).
- FFmpeg y FFprobe disponibles en `PATH`. FFprobe se usa durante la inspección;
  FFmpeg decodifica frames para la detección y genera videos de prueba.

Desde la carpeta del repositorio:

```bash
uv sync --locked
export PYTHONPATH=src
```

En PowerShell, usar `$env:PYTHONPATH = "src"` en lugar de `export`. El proyecto
usa `.venv`, NumPy y OpenCV headless, con versiones fijadas en `uv.lock`. Al no tener todavía un
backend de empaquetado, `PYTHONPATH=src` permite ejecutar el paquete desde el checkout.

## Directorio del usuario

La opción global `--root` define dónde se guarda el trabajo. También se puede
usar `CONTENT_AGENT_HOME`; el valor por defecto es `~/.content-agent-orchestrator`.
El contenido del usuario queda fuera del checkout:

```text
<root>/
├── artifacts/
│   ├── raw_footages/                      # videos fuente
│   ├── runs/<run_id>/<node>/<attempt>/    # resultados intermedios privados
│   ├── identity_registry/<sha256>.json    # snapshots biométricos privados
│   ├── virtual_candidates/<hash>/        # rostros B pendientes de revisión
│   ├── virtual_faces/<virtual_id>/        # retratos B aprobados
│   └── virtual_registry/<sha256>.zip     # catálogos con píxeles de B
├── temps/<run_id>/<node>/                 # trabajo recuperable
├── outputs/<run_id>/                     # archivos del único nodo final
└── state/
    ├── executions.sqlite                 # ejecuciones y checkpoints
    ├── identities.sqlite                 # personas y referencias persistentes
    └── locks/                            # exclusión de una misma ejecución
```

Copiar los videos originales a `artifacts/raw_footages/`, o indicar una ruta
absoluta a un archivo externo. **Todas las rutas relativas, tanto de inputs como
del JSON del pipeline, se interpretan respecto del directorio de trabajo**.
`--root` define el almacenamiento de resultados y estado; no cambia la base de
las rutas de inputs. La CLI crea las carpetas de almacenamiento que falten.
No admite inputs procedentes de `temps/` ni de `state/`.

## Inspeccionar un video

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  inspect /ruta/al/directorio-del-usuario/artifacts/raw_footages/clip.mp4
```

El pipeline integrado ejecuta `video.inspect` y luego `report.export`. Devuelve
JSON por stdout con el ID de ejecución, los resultados y sus hashes; los eventos
de progreso y errores se escriben en stderr. Los errores devuelven código 1 y
una interrupción de teclado devuelve 130.

La inspección no recodifica ni modifica el input. Produce:

- Un manifiesto con resolución, streams de video/audio, orientación cuando está
  disponible, información de color, duración, FPS declarados y time base.
- Un artifact JSONL con el índice de cada frame, su timestamp entero en unidades
  del time base y el tiempo de presentación en segundos.
- Una clasificación de timing `constant`, `variable` o `unknown`, calculada a
  partir de los frames decodificados, no solamente del FPS declarado. La
  tolerancia usada queda en el informe: dos ticks del time base o dos
  microsegundos, lo que resulte mayor.

El informe final queda en `outputs/<run_id>/report.json`; el manifiesto y los
timestamps intermedios quedan en `artifacts/runs/`. Los tags libres del video y
la ruta original no se incluyen en el informe. La inspección recorre el stream
de video completo; escribe los timestamps a disco y los procesa por líneas para
no acumular todos los frames en memoria. No carga imágenes completas en Python.

## Ejecutar un pipeline declarado

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  run examples/inspect-video.json \
  --input video=/ruta/al/directorio-del-usuario/artifacts/raw_footages/clip.mp4
```

El ejemplo tiene esta estructura:

```json
{
  "nodes": [
    {
      "id": "inspect",
      "module": "video.inspect",
      "inputs": {"video": {"external": "video"}},
      "parameters": {"timeout_seconds": 600}
    },
    {
      "id": "export",
      "module": "report.export",
      "inputs": {
        "inspection": {"node": "inspect", "output": "manifest"}
      }
    }
  ],
  "final": "export"
}
```

Se rechazan ciclos, módulos desconocidos, IDs duplicados, outputs inexistentes,
conexiones entre tipos incompatibles y nodos que no conduzcan al único final.
El orden del JSON no tiene que coincidir con el orden de ejecución. El ejecutor
es secuencial; permite ramas convergentes, pero todavía no ejecuta en paralelo.
`timeout_seconds` limita cada llamada a FFprobe y puede ajustarse entre 1 y
86400 segundos para videos largos.

## Detectar y seguir rostros

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  analyze-faces /ruta/al/directorio-del-usuario/artifacts/raw_footages/clip.mp4
```

Ejecuta `video.inspect → face.detect → face.track → face.report`, con un único
nodo final. YuNet está incluido con su licencia MIT y checksum verificado; el
procesamiento usa CPU y no requiere descargas ni servicios externos. El detector
trabaja con frames reducidos en memoria y devuelve coordenadas del video original,
incluidas rotaciones de cámara de 90°, 180° y 270°. El archivo fuente permanece intacto.

Los outputs son un informe JSON y archivos JSONL con trayectorias por frame,
resúmenes de tracks e intervalos sin detecciones o con asociaciones ambiguas.
Los `track_id` tienen alcance dentro de un clip; todavía no identifican personas
entre videos por sí solos. **Este comando analiza rostros; no produce un video anonimizado.**

Se puede ajustar `--score-threshold` (por defecto 0.85) y `--max-dimension`
(por defecto 640). Para configurar tracking o checkpoints, usar
[examples/analyze-faces.json](examples/analyze-faces.json). Los contratos,
parámetros, outputs y recuperación se describen en
[docs/face-analysis.md](docs/face-analysis.md).

## Reconocer personas entre videos

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario recognize-faces /ruta/clip.mp4
```

Agrega `face.embed → identity.resolve → identity.report` al análisis. SFace está
incluido con licencia Apache 2.0 y checksum verificado. Cada observación recibe
un estado conocido, desconocido, ambiguo o inutilizable; no se propaga una
identidad automáticamente a todo un track. Los outputs contienen metadatos,
sin vectores ni imágenes. Los embeddings son artifacts intermedios privados.

El alta en SQLite se realiza mediante `identity register`, usando las rutas
`private_intermediates` devueltas por la CLI y un track elegido explícitamente.
`identity list`, `identity snapshot` e `identity delete` permiten administrar
el registro. El reconocimiento usa snapshots inmutables: los cambios del
registro invalidan la caché de resolución. Consultar el flujo completo,
parámetros, eliminación y límites de privacidad en
[docs/identity-recognition.md](docs/identity-recognition.md).

## Crear y reutilizar un rostro virtual

`virtual create <identity_id> --profile /ruta/perfil.json` crea un candidato B
para una persona A registrada, usando SDXL local o un generador local
externo. El perfil define género, tono de piel, edad adulta y preferencias como
maquillaje o delineado. `virtual prepare-model` descarga y verifica
un checkpoint fotorealista RealVisXL o SDXL base. Crear B usa esos pesos
localmente, sin API de imágenes ni transmisión del rostro A.

Después de revisar la imagen, `virtual approve <identity_id> <candidate_id>
--reviewed` asocia B de forma inmutable. Se validan geometría y separación
biométrica respecto de las personas registradas. Repetir la creación reutiliza
el mismo PNG: mantener la identidad no depende de regenerar el mismo prompt.

`assign-virtual-faces /ruta/clip.mp4` extiende el DAG con `virtual.resolve →
virtual.report`, conservando un único final. Publica asignaciones por frame y
las caras sintéticas seleccionadas, sin intercambiar todavía el video. Casos
desconocidos o ambiguos permanecen sin target. Consultar comandos, revisión
visual, caché, backend local y límites en
[docs/virtual-identities.md](docs/virtual-identities.md).

## Caché, fallos y reanudación

`--cache reuse` es el modo por defecto. Los resultados completos se pueden
reutilizar entre ejecuciones si coinciden el nombre y versión del módulo, los
parámetros y los hashes/tipos de sus inputs. Antes de reutilizarlos se comprueba
que **todos** sus archivos existan y coincidan en tamaño y SHA-256. Si no,
el módulo se vuelve a ejecutar. Cada nueva ejecución conserva sus propias
copias de los resultados y no modifica resultados anteriores.

Si un módulo falla, se detiene el pipeline y se conserva su workspace temporal.
Los temporales de los módulos que ya guardaron sus resultados se eliminan.
El error y el evento inicial identifican la ejecución; para reanudarla:

```bash
uv run --locked python -m content_agent_orchestrator \
  --root /ruta/al/directorio-del-usuario \
  inspect /ruta/al/directorio-del-usuario/artifacts/raw_footages/clip.mp4 \
  --resume <run_id> --cache reuse
```

La reanudación requiere el mismo pipeline, rutas y versiones. Conserva el
workspace fallido solamente si sus inputs siguen coincidiendo. Cada módulo es
responsable de interpretar su caché interna; la inspección actualmente repite
las llamadas a FFprobe tras un fallo. Los archivos de `result/` incompletos
siempre se descartan y nunca cuentan como resultados exitosos. `face.detect`
reutiliza bloques completos con checksum verificado; al reanudar vuelve a
decodificar desde el principio para conservar el orden, pero omite la inferencia
de los bloques ya confirmados. `face.track` y los informes reinician su pasada
de metadatos si fallan.

`--cache fresh` desactiva la reutilización. Combinado con `--resume`, elimina y
recrea los temporales de los nodos de **esa ejecución**, sin borrar el trabajo
de otras ejecuciones ni los artifacts anteriores. Una ejecución terminada no
se reanuda: ejecutar nuevamente el comando crea otro ID.

Los resultados se copian a un staging temporal, se verifican y se publican con
un rename del directorio completo. `outputs/<run_id>/.publication.json` es un
registro interno para recuperar una interrupción entre ese rename y el commit
de SQLite. Un destino preexistente que no corresponda a una publicación íntegra
de esa ejecución se conserva y provoca un error. Para el rename atómico, las
carpetas de almacenamiento deben estar en el mismo filesystem.

## Agregar un módulo

Implementar el protocolo `Module` de `contracts.py` y registrarlo en
`builtin_registry()` (o pasar un registro propio a `Runner`):

- `name` y `version` identifican la implementación.
- `input_kinds` y `output_kinds` declaran nombres y tipos de artifacts.
- `validate_parameters()` rechaza configuración inválida antes de ejecutar.
- `execute()` recibe inputs, parámetros y un `ModuleContext`.
- Crear archivos de trabajo/caché en `context.work_dir` y resultados completos
  en `context.result_dir`; devolver un mapa de nombre de output a archivo.

No modificar los inputs. El runner comprueba sus hashes antes y después de
ejecutar, pero **los módulos son código de confianza, no procesos aislados**.
Los resultados deben depender del contenido de los inputs, los parámetros y
la versión: incluir referencias a versiones de modelos y semillas en la
configuración, y actualizar `version` cuando cambie el comportamiento. No poner
credenciales en los parámetros, ya que la configuración se guarda en SQLite.

El módulo no decide entre `artifacts/` y `outputs/`: lo decide su posición en el
DAG. Un módulo final puede devolver varios archivos, pero el nodo final es uno.

## Pruebas

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check --no-cache .
uv run --locked ruff format --check --no-cache .
```

Las pruebas cubren ramas convergentes, validación del DAG, caché e integridad,
fallos y recuperación, publicación incompleta, protección de rutas y CLI.
Las de integración generan videos pequeños con audio y timing constante o
variable, y comprueban que el input no cambie. Sin FFmpeg/FFprobe esas pruebas
se marcan como omitidas; una validación completa requiere tenerlos instalados.

## Alcance siguiente

Los siguientes módulos serán reemplazo facial, validación visual, corrección conservadora y exportación
verificada. El intercambio de voz podrá incorporarse después como otro módulo.
El análisis usa CPU, con modelos incluidos, y no envía videos a servicios externos.
La generación de B usa SDXL local o un generador externo local.
Los inputs, temporales, registro y embeddings deben tratarse como material
sensible; el directorio del usuario debe estar en almacenamiento privado.
