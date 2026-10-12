# Identidades persistentes

El pipeline integrado es `video.inspect → face.detect → face.track → face.embed
→ identity.resolve → identity.report`. Tiene un único nodo final y conserva el
video original. **Todavía no produce un video anonimizado.** No realiza altas
automáticas: un `track_id` describe continuidad geométrica, mientras que un
`identity_id` UUID identifica una persona registrada explícitamente entre clips.

## Primera ejecución y alta

```bash
export PYTHONPATH=src
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  recognize-faces /datos/influencer/artifacts/raw_footages/clip.mp4
```

Con un registro vacío, las observaciones utilizables serán `unknown`. Revisar
los tracks del informe y elegir uno que corresponda a la persona que se desea
registrar. La respuesta JSON de la CLI incluye `private_intermediates.frames`
y `private_intermediates.summary`: son rutas a los artifacts de embeddings y
su manifiesto, necesarios para el alta. No imprime los vectores.

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  identity register --embeddings /ruta/devuelta/frames.jsonl \
  --summary /ruta/devuelta/summary.json --track track-000001 --alias creador
```

El alias es opcional y único; conviene usar un seudónimo. El alta devuelve un
`identity_id` persistente. Requiere al menos tres frames con embeddings
utilizables y rechaza asociaciones ambiguas o referencias incompatibles dentro
del track. Compara cada observación utilizable con la primera y con las
referencias retenidas, usando similitud coseno mínima 0.65, configurable con
`--min-similarity` entre 0.5 y 1. Es una comprobación de coherencia, no una
certificación de identidad ni de calidad biométrica.

Se conservan como máximo 16 referencias por alta: las primeras 15 y la última
observación, o todas si hay menos. Para agregar referencias de otro clip a la
misma persona, usar `--identity-id <UUID>` en lugar de `--alias`; todas las
observaciones utilizables del track deben ser compatibles con todas las
referencias existentes. El límite total es 32. Repetir un alta de los mismos
frames sobre el mismo ID no duplica referencias ni aumenta la revisión.
Los errores revierten el alta completa mediante una transacción SQLite.

## Reconocer otros clips

Volver a ejecutar `recognize-faces` sobre otro video. Cada observación recibe:

| Estado | Significado |
| --- | --- |
| `known` | Mejor coincidencia supera el umbral y el margen respecto de otras personas. |
| `unknown` | Registro vacío o mejor coincidencia por debajo del umbral. |
| `ambiguous` | Puntuaciones cercanas entre personas o asociación geométrica ambigua. |
| `unusable` | Rostro pequeño, confianza baja o fallo de alineación. |

Por defecto, `--min-similarity` vale 0.6 y `--ambiguity-margin` vale 0.12. Se
calcula la mayor similitud coseno entre la observación y las referencias de
cada persona, y se compara la mejor persona con la segunda. Los scores no son
probabilidades. Estos valores son conservadores respecto del ejemplo SFace
original, pero requieren calibración con clips representativos; no garantizan
ausencia de falsos positivos. No inferir una identidad para observaciones
desconocidas, ambiguas o inutilizables.

El informe final contiene `report.json`, `frames.jsonl` y `tracks.jsonl`, sin
vectores ni imágenes. Un track será `mixed` si aparecen coincidencias con más
de una identidad y registrará los cambios de identidad por frame. Será
`uncertain` si combina coincidencias conocidas con observaciones dudosas.
La asignación de un frame no se propaga a los demás frames del track.

## Estado, snapshots y caché

El registro está en `<root>/state/identities.sqlite`. Contiene referencias,
IDs, aliases y procedencia (hash del video, track y frame). Solamente los
comandos explícitos `identity register` y `identity delete` modifican personas.
Inicializar el registro vacío no inscribe ninguna persona.

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer identity list
uv run --locked python -m content_agent_orchestrator --root /datos/influencer identity snapshot
```

`list` muestra IDs, aliases y cantidades; `snapshot` devuelve la ruta de una
copia inmutable en `artifacts/identity_registry/<sha256>.json`. Esta carpeta
está reservada para snapshots administrados. Contienen referencias biométricas
privadas. Un mismo registro/revisión produce exactamente el mismo archivo.
La creación de la copia usa una transacción, por lo que no mezcla lecturas de
distintas revisiones.

`recognize-faces` toma un snapshot del registro al comenzar. Su hash es un
input explícito de `identity.resolve`: agregar o borrar referencias invalida
la caché de resolución e informe, mientras que puede reutilizar inspección,
detección, tracking y extracción. Los módulos no consultan una DB mutable.
La revisión y hash usados quedan en el informe.

Para reanudar después de cambios del registro, usar la ruta `registry_snapshot`
devuelta en la ejecución original, junto con la misma configuración:

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  recognize-faces /ruta/clip.mp4 --resume <run_id> --snapshot /ruta/snapshot-original.json
```

Esto resuelve contra el registro histórico elegido. Una nueva ejecución sin
`--snapshot` usa el registro actual. Si el snapshot original fue borrado,
iniciar una ejecución nueva. `face.embed` conserva temporales al fallar, pero
reinicia la extracción completa al reanudar; no reutiliza embeddings parciales.
`--cache fresh` tiene el mismo alcance por ejecución que en los otros pipelines.

## Modelos y contratos componibles

SFace float32 está incluido (~37 MiB), con licencia Apache 2.0, commit de origen
y checksum verificado antes de cargarlo. YuNet y SFace trabajan en CPU con
OpenCV headless; no requieren nuevas dependencias ni servicios externos.
`--detector` y `--recognizer` permiten indicar otras copias de esos mismos
modelos verificados, no modelos arbitrarios. No se descarga nada al ejecutar.

`face.embed` recibe el video, manifiesto, timestamps, frames/summary de tracking
y modelo `face.recognizer.sface`. Verifica procedencia, geometry y timestamps,
decodifica en resolución original sin autorrotación y alinea con los cinco
landmarks en coordenadas del video codificado. No persiste crops. Produce
`face.embeddings` JSONL con un vector normalizado de 128 números por observación
utilizable y `face.embedding-summary`. Mantiene los timestamps enteros y
los tiempos originales, incluidos videos VFR. Trabaja con un frame en memoria;
el espacio en disco crece con frames y rostros.

Los parámetros de extracción son `min_face_pixels=64`, `min_confidence=0.9`
y `timeout_seconds=600` (ajustar para videos largos). Los filtros de tamaño y
confianza son aproximaciones: no verifican iluminación, pose, oclusión o nitidez.
El modelo y la versión de alineación están ligados a los vectores, snapshots
y registro; mezclar modelos o preprocessing se rechaza.

`identity.resolve` recibe `face.embeddings`, `face.embedding-summary` y
`identity.registry-snapshot`, y devuelve `identity.observations`,
`identity.tracks`, `identity.resolution-summary`. `identity.report` exporta los
metadatos y es el único final del pipeline integrado.
El ejemplo [../examples/recognize-faces.json](../examples/recognize-faces.json)
permite configurar cada etapa y usar el comando genérico `run` con inputs
`video`, `detector`, `recognizer` y `registry` (snapshot previamente creado).

## Privacidad y eliminación

Los embeddings permiten reconocer a una persona y deben tratarse como datos
sensibles. Permanecen en `artifacts/runs/`, fuera de los outputs finales.
La DB y snapshots nuevos tienen permisos 0600 en POSIX; todos los artifacts
publicados nuevos también se crean con 0600. En Windows, proteger el directorio
mediante ACL. No hay cifrado integrado: usar almacenamiento privado y cifrado
de disco, especialmente en la nube. No subir el directorio del usuario al repo.
Los reportes también son privados porque enlazan apariciones entre videos.

```bash
uv run --locked python -m content_agent_orchestrator --root /datos/influencer \
  identity delete <UUID>
```

Elimina la identidad y sus referencias de SQLite con `secure_delete`, aumenta
la revisión y purga **todos** los snapshots administrados, incluso corruptos,
porque pueden contener referencias antiguas. No garantiza borrado físico en
SSD, journals del filesystem o backups. Conserva videos originales, artifacts
de embeddings, temporales fallidos, informes y copias externas de snapshots;
la CLI informa ese alcance. Si se requiere borrar todo material biométrico,
eliminar también esas ejecuciones y sus backups según la política propia.
La eliminación de una identidad no anonimizó los videos ni sus copias.
