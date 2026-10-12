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

## Crear un candidato en la nube

Con A registrado mediante `identity register`, ejecutar:

```bash
export PYTHONPATH=src
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  virtual create <identity_id_A> --profile /ruta/perfil.json
```

Requiere `OPENAI_API_KEY` en el entorno, acceso a la API de imágenes de OpenAI
y conectividad HTTPS a `api.openai.com`. No lee archivos `.env`. El modelo por
defecto es `gpt-image-1`; `--model` también permite `gpt-image-1.5` y
`gpt-image-1-mini`. Solicita una imagen PNG opaca de 1024×1024 y calidad alta.
La API puede tener costo y requisitos de acceso según la cuenta. **Solamente
envía el prompt del perfil: no envía A, videos, aliases, IDs ni embeddings.**

El cliente usa la biblioteca estándar, rechaza redirects, limita la respuesta,
acepta bytes base64 y nunca descarga URLs retornadas por el proveedor. Los
errores no muestran claves, cuerpos de respuestas ni headers. No hay reintentos
automáticos de solicitudes cobrables; un timeout puede ocurrir después de que
el proveedor haya procesado la solicitud.

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
adaptador a un servidor ComfyUI o a un modelo local). Este PR no instala modelos
de difusión, CUDA ni un servidor local; no valida consumo de VRAM de un modelo
particular. La aplicación no agrega dependencias Python.

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
reutiliza la imagen completa; no vuelve a cobrar su generación. Un checkpoint
incompleto o con checksum incorrecto no se reutiliza. Si falló la conexión
antes de guardar una respuesta completa, ejecutar de nuevo puede cobrar otra
solicitud; no es posible recuperar ese resultado desde esta aplicación.

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
reconocimiento. Para reanudar una ejecución tras cambios, indicar tanto
`--snapshot /ruta/registro-original.json` como `--catalog /ruta/catalogo-original.zip`,
con la misma configuración. Las dos rutas aparecen en la respuesta de la CLI.
La ejecución resolverá contra ese estado histórico.

Los vínculos A→B son privados: candidatos, retratos y catálogos se crean con
permisos 0600 en POSIX. `identity delete A` elimina la asociación y el PNG de
SQLite por cascade y purga snapshots administrados de ambos registros. Conserva
candidatos, exportaciones de retratos, outputs históricos y backups. Un ZIP
histórico todavía puede vincular A con B: aplicar la propia política de
eliminación a todas sus copias.
