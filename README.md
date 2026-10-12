# Content Agent Orchestrator

Coordinador de pipelines de módulos componibles. Cada pipeline es un DAG que
converge en **un único nodo final**. Esta primera entrega permite inspeccionar
videos y exportar informes; todavía no implementa reconocimiento de personas,
generación de identidades, intercambio facial ni validación de anonimización.

## Requisitos y preparación

- Python 3.12, según `.python-version`.
- [uv](https://docs.astral.sh/uv/).
- FFmpeg y FFprobe disponibles en `PATH`. FFprobe se usa durante la inspección;
  FFmpeg se usa para generar los videos de las pruebas de integración.

Desde la carpeta del repositorio:

```bash
uv sync --locked
export PYTHONPATH=src
```

En PowerShell, usar `$env:PYTHONPATH = "src"` en lugar de `export`. El proyecto
usa `.venv` y no incorpora nuevas dependencias de Python. Al no tener todavía un
backend de empaquetado, `PYTHONPATH=src` permite ejecutar el paquete desde el checkout.

## Directorio del usuario

La opción global `--root` define dónde se guarda el trabajo. También se puede
usar `CONTENT_AGENT_HOME`; el valor por defecto es `~/.content-agent-orchestrator`.
El contenido del usuario queda fuera del checkout:

```text
<root>/
├── artifacts/
│   ├── raw_footages/                      # videos fuente
│   └── runs/<run_id>/<node>/<attempt>/    # resultados intermedios
├── temps/<run_id>/<node>/                 # trabajo recuperable
├── outputs/<run_id>/                     # archivos del único nodo final
└── state/
    ├── executions.sqlite                 # ejecuciones y checkpoints
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
siempre se descartan y nunca cuentan como resultados exitosos.

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

Los siguientes módulos serán detección/seguimiento, resolución de identidades
en una base persistente, creación/recuperación de identidad virtual,
reemplazo facial, validación visual, corrección conservadora y exportación
verificada. El intercambio de voz podrá incorporarse después como otro módulo.
Esta entrega no usa GPU, descarga modelos ni envía videos a servicios externos.
Los inputs, temporales y futuros embeddings deben tratarse como material
sensible; el directorio del usuario debe estar en almacenamiento privado.
