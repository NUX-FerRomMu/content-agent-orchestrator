# Rostros B persistentes

Esta etapa crea un rostro ficticio B y lo asocia a un `identity_id` A ya
registrado. Guarda los **píxeles exactos** de B: repetir un prompt o una semilla
no garantiza reproducir un rostro, así que la asociación aprobada es inmutable.
Los videos fuente permanecen intactos. Todavía no hay intercambio facial ni
validación de anonimización del video.

## Perfil explícito

Usar [../examples/virtual-profile.json](../examples/virtual-profile.json) como
plantilla, con atributos que correspondan a A:

```json
{
  "gender": "woman",
  "skin_tone": 6,
  "undertone": "warm",
  "age_years": 35,
  "features": "Subtle eyeliner and natural makeup"
}
```

`gender` acepta `woman`, `man` o `androgynous`; `skin_tone` es una selección
manual de la escala Monk del 1 al 10; `undertone` acepta `warm`, `neutral` o
`cool`. No se infieren género ni tono de piel mediante embeddings. La edad
virtual debe estar entre 25 y 65 años; `features` contiene hasta 1000 caracteres
de preferencias cosméticas. El programa añade instrucciones de persona
ficticia adulta, sin anteojos ni rasgos excéntricos, frontal, bien iluminada,
sin oclusiones y con expresión neutral. El texto libre se trata como preferencias
subordinadas a esas condiciones.

## Preparar el generador local

El backend predeterminado es SDXL local. `realvis-xl` selecciona
[RealVisXL V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0), un checkpoint
SDXL orientado al fotorealismo, útil para retratos. No es un modelo exclusivo de
caras. `sdxl-base` permite usar el
[SDXL 1.0 original](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
Los modelos puramente faciales como StyleGAN3 no se integran: no proporcionan
el mismo control por prompt y su licencia limita el uso comercial.

La generación necesita PyTorch, Diffusers, Transformers, Accelerate y
Safetensors, fijados en `uv.lock` dentro del grupo opcional `generation`:

```bash
uv sync --locked --group generation
```

La inspección, el reconocimiento y la asignación de B ya aprobadas mantienen
el entorno liviano (`uv sync --locked`, sin el grupo de generación). Para
conservar el grupo al ejecutar comandos, usar `uv run --locked --group generation`
o ejecutar `.venv/bin/python` después de instalarlo. El runtime importa estas
bibliotecas solamente al generar una imagen nueva.

La preparación es una descarga explícita:

```bash
export PYTHONPATH=src
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  virtual prepare-model --model realvis-xl
```

`--revision` permite elegir una revisión. `main` se resuelve a un commit
inmutable en el momento de preparar; no se consulta durante la generación.
Antes de descargar se comprueba que el repositorio declare la licencia
CreativeML Open RAIL++-M (`openrail++`) y que no requiera acceso cerrado.
Si no puede verificarse esa licencia, se rechaza la preparación. La etiqueta
especifica una licencia con restricciones de uso; no equivale a dominio público.
La copia conserva README y archivos de licencia publicados para revisión.

Se guardan solamente configuración, tokenizadores y pesos `safetensors`.
Cada archivo se verifica contra SHA-256 LFS o hash Git publicado por el
repositorio. Un manifiesto registra también SHA-256 locales, tamaños y commit.
Los archivos completos se reutilizan tras un fallo; descargas parciales se
conservan como diagnóstico y se reinician. Un modelo incompleto no se publica.
La descarga se limita a 24 GiB y el directorio completo se publica con rename.
Si la copia publicada no supera la verificación, repetir `virtual prepare-model`
descarga un reemplazo en temporales y lo verifica antes de sustituirla. Una
descarga fallida conserva la copia anterior y los archivos completos para
reintentar; un fallo al publicar restaura la copia anterior.

El resultado devuelve `model_dir` bajo
`artifacts/models/<modelo>/<commit>/`. Los pesos grandes quedan en el directorio
del usuario y nunca se agregan al repositorio. El runtime rechaza archivos
sin verificar, symlinks y formatos que requieran pickle o código remoto.
La preparación requiere HTTPS a Hugging Face y a los hosts de descarga
indicados por el proveedor; no necesita una clave de API de imágenes.

## Crear B sin servicios de generación

Con A registrado mediante `identity register`, ejecutar:

```bash
uv run --locked --group generation python -m content_agent_orchestrator --root /datos/influencer \
  virtual create <identity_id_A> --profile /ruta/perfil.json \
  --model-dir /ruta/devuelta/por/prepare-model
```

La generación usa exclusivamente esos archivos locales: `local_files_only=True`
y `use_safetensors=True`. No hay backend de OpenAI, no se lee `OPENAI_API_KEY`
y no se envían prompts, A, videos ni embeddings a un proveedor de imágenes.
Puede ejecutarse en una PC o en una máquina de nube con GPU NVIDIA compatible.

La configuración inicial usa CUDA FP16, 1024×1024, 30 pasos, guidance 5 y
DPM-Solver++ con Karras. VAE slicing/tiling y CPU offload están habilitados
para reducir VRAM. `--seed`, `--steps` y `--offload/--no-offload` permiten ajustar
la ejecución; se registran en la procedencia y la clave del job. Apunta a una
GPU de 16 GB, pero este entorno no tiene GPU disponible para medir su consumo.
`--device cpu` es un fallback explícito FP32 mucho más lento y con alto uso de RAM.

SDXL recibe un prompt compacto con los atributos requeridos antes de las
preferencias cosméticas y un prompt negativo contra menores, anteojos,
oclusiones y deformaciones. Se comprueba el límite de **ambos** tokenizadores
CLIP; un texto demasiado largo se rechaza antes de inferir, en lugar de truncar
silenciosamente atributos. Acortar `features` si excede ese límite.

El resultado incluye `candidate_id`, `portrait_path` y
`status: needs_visual_review`. El candidato queda en
`artifacts/virtual_candidates/<candidate_id>/`; su manifiesto privado contiene
perfil, procedencia del generador, hashes, calidad y embedding de B.

## Validación y asociación

Antes de publicar un candidato, se comprueba localmente:

- PNG opaco RGB de 8 bits, entre 256 y 2048 píxeles por dimensión, hasta 16 MiB.
- Exactamente un rostro YuNet detectado, confianza ≥0.9, cara completa y de
  al menos 128 píxeles; landmarks dentro de la imagen.
- Alineación aproximadamente frontal: roll ≤10°, posición de nariz respecto
  de los ojos, ojos separados al menos 40 píxeles y nitidez mínima del crop.
- Embedding SFace válido, con similitud coseno **menor que 0.3 contra todas
  las referencias reales registradas**, no solamente A.

Estas son comprobaciones estructurales aproximadas. No certifican edad,
ausencia de anteojos, género ni tono de piel. Abrir `portrait_path`, revisar
que la imagen sea claramente adulta, sin anteojos ni rasgos excéntricos, que
cumpla el perfil de A y que su procedencia sea generación ficticia. Después:

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  virtual approve <identity_id_A> <candidate_id> --reviewed
```

`--reviewed` registra esa revisión visual; es necesario para asignar B, no se
activa por defecto. La aprobación vuelve a calcular los checks y la separación
contra el registro actual. Sin aprobación, ningún pipeline usará el candidato.
El programa no puede demostrar que una cara generada no se parezca a alguna
persona real en el mundo; exige procedencia ficticia y evita coincidencias con
las personas conocidas por este registro.

Una vez aprobado, A recibe un `virtual_id` estable. SQLite guarda PNG, hash,
perfil, generador, validación y fecha de revisión. Se exporta una copia a
`artifacts/virtual_faces/<virtual_id>/portrait.png`. Repetir `create` con el
mismo perfil reutiliza B sin volver a generar, incluso cambiando el backend.
Cambiar el perfil o intentar aprobar otra B sobre A provoca un error: no se
reemplaza silenciosamente la identidad del influencer.

## Generador local

`--backend command --generator /ruta/generador.json` ejecuta un generador externo
de confianza, instalado por el usuario en su propio entorno (por ejemplo, un
adaptador a un servidor ComfyUI o a un modelo local). Este adaptador permite usar otro entorno o servidor. No instala sus
dependencias ni ejecuta un servidor automáticamente.

El JSON configura el comando, identificación/versionado del modelo y semilla:

```json
{
  "argv": ["/ruta/al/python-del-generador", "/ruta/mi_generador.py"],
  "model": "nombre-del-modelo-version-y-checksum",
  "seed": 42
}
```

Se ejecuta sin shell y recibe por stdin un JSON con `prompt`, `size: [1024,
1024]`, `seed` y `output_path`. Debe escribir el PNG en `output_path` y terminar
con código 0. La semilla es un entero uint32. La configuración y el comando son
código de confianza, no un sandbox: no incorporar comandos recibidos de terceros
ni claves en `argv`. Usar rutas absolutas. El proceso tiene timeout de 600 s;
stdout y stderr se descartan para no guardar secretos de scripts externos.
Se conservan los archivos que el generador haya creado en su workspace al fallar.

## Temporales y recuperación

Los jobs usan `temps/virtual_generation/<hash-del-pedido>/` con lock por pedido.
Después de recibir la imagen, se guarda PNG y checkpoint con checksum antes
de validar. Si la validación falla o se interrumpe, otro `create` idéntico
reutiliza la imagen completa; no vuelve a ejecutar la difusión. Un checkpoint
incompleto o con checksum incorrecto no se reutiliza. Si falló el generador
antes de guardar una imagen completa, ejecutar de nuevo reinicia la inferencia.

`virtual create ... --cache fresh` elimina solamente los temporales del pedido
correspondiente y permite volver a intentar una generación fallida. No borra
candidatos ya publicados, B aprobadas ni otras ejecuciones. Después de publicar
un candidato válido, sus temporales se eliminan.

## Usar B en el DAG

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer virtual list
uv run --locked python -m content_agent_orchestrator --root /datos/influencer virtual snapshot
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  assign-virtual-faces /ruta/clip.mp4
```

El pipeline es `inspect → detect → track → embed → identity.resolve →
virtual.resolve → virtual.report`, con un único final. La creación/aprobación
de identidades es un comando explícito fuera de los módulos cacheables, igual
que el alta de personas; el DAG usa snapshots de ese estado.

El catálogo `artifacts/virtual_registry/<sha256>.zip` contiene un manifiesto y
los PNG exactos de las B aprobadas, sin embeddings reales ni sintéticos. Usa
revisión independiente de identidades virtuales y la revisión del registro
fuente usada al volver a validar su separación. El hash cubre los píxeles;
guardar solamente rutas permitiría reutilizar decisiones sobre imágenes cambiadas.
Antes de crear un snapshot, cada B se vuelve a validar contra todas las
referencias reales actuales. Si se incorpora una referencia parecida a B,
el programa falla en lugar de emitir un catálogo inseguro.

`virtual.resolve` recibe `identity.observations`, `identity.resolution-summary`
y `virtual.catalog`. El catálogo y la resolución de A deben compartir UUID
y revisión del registro fuente. Devuelve `virtual.observations`,
`virtual.resolution-summary` y `virtual.targets`, un ZIP con **solamente** las
B utilizadas por ese clip. Cada observación queda como `ready`, `missing_target`
o `unresolved_source`. Nunca usa el candidato más parecido de una identidad
desconocida ni asigna una B por defecto. Los estados sin target quedan explícitos
para la futura corrección conservadora del video.

`virtual.report` es el único final integrado y publica `report.json`,
`frames.jsonl` y `targets.zip`. Estos outputs contienen las caras sintéticas
seleccionadas y sus asociaciones, pero ninguna imagen de A ni embeddings.
Para agregar un futuro módulo de intercambio, usar los outputs de
`virtual.resolve` como artifacts intermedios y converger en ese nuevo final.
El ejemplo [../examples/assign-virtual-faces.json](../examples/assign-virtual-faces.json)
usa inputs `video`, `detector`, `recognizer`, `registry` y `catalog` en el comando
genérico `run`.

Crear/aprobar B invalida los módulos virtuales, conservando caché de
reconocimiento. Para reanudar cualquier ejecución, indicar tanto
`--snapshot /ruta/registro-original.json` como `--catalog /ruta/catalogo-original.zip`,
con la misma configuración. Las dos rutas aparecen en la respuesta de la CLI.
Si falta alguna, la CLI rechaza la reanudación antes de crear nuevos snapshots.
La ejecución resolverá contra ese estado histórico.

Los vínculos A→B son privados: candidatos, retratos y catálogos se crean con
permisos 0600 en POSIX. `identity delete A` elimina la asociación y el PNG de
SQLite por cascade y purga snapshots administrados de ambos registros. Conserva
candidatos, exportaciones de retratos, outputs históricos y backups. Un ZIP
histórico todavía puede vincular A con B: aplicar la propia política de
eliminación a todas sus copias.
