"""Saved input bindings, private imports and discoverable execution recovery."""

import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .face_data import iter_rows, read_json
from .identity_registry import IdentityRegistry
from .pipeline import Pipeline
from .runner import Runner
from .storage import UserStorage, digest_file
from .swap_models import prepared_swap_paths
from .virtual_face import appearance_profile
from .virtual_registry import VirtualRegistry


def save_document(path: Path, document: dict) -> Path:
    """Write a private JSON document exclusively, without truncating prior work."""
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def prompt(label: str, default: str | None = None, choices=()) -> str:
    """Keep prompts on stderr so stdout remains machine-readable JSON."""
    while True:
        suffix = f" [{default}]" if default is not None else ""
        options = f" ({'/'.join(choices)})" if choices else ""
        print(f"{label}{options}{suffix}: ", end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip() or default
        except EOFError:
            raise PipelineError(
                "Assistant cancelled: input ended; no job was saved"
            ) from None
        if answer is not None and answer != "" and (not choices or answer in choices):
            return answer
        print(
            "Enter one of the listed values." if choices else "A value is required.",
            file=sys.stderr,
        )


def create_profile(args) -> dict:
    """Gather an explicit source appearance profile; infer no demographics."""
    if args.interactive:
        args.gender = prompt(
            "Presentación de género de A", args.gender, ("woman", "man", "androgynous")
        )
        args.skin_tone = int(
            prompt(
                "Tono de piel Monk (1 claro – 10 oscuro)",
                str(args.skin_tone) if args.skin_tone else None,
                tuple(str(i) for i in range(1, 11)),
            )
        )
        args.undertone = prompt("Subtono", args.undertone, ("neutral", "warm", "cool"))
        args.age_years = int(
            prompt(
                "Edad adulta de B",
                str(args.age_years),
                tuple(str(i) for i in range(25, 66)),
            )
        )
        print(
            "Preferencias cosméticas (Enter para ninguna): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        try:
            args.features = input().strip()
        except EOFError:
            raise PipelineError("Assistant cancelled; no profile saved") from None
    profile = appearance_profile(
        {
            "gender": args.gender,
            "skin_tone": args.skin_tone,
            "undertone": args.undertone,
            "age_years": args.age_years,
            "features": args.features,
        }
    )
    return {
        "profile_path": str(save_document(args.output, profile)),
        "profile": profile,
    }


def import_video(storage: UserStorage, supplied: Path) -> dict:
    """Copy a source into raw_footages, verifying content and never overwriting."""
    source = supplied.expanduser().resolve()
    if not source.is_file():
        raise PipelineError("Import requires an existing video file")
    if any(source.is_relative_to(storage.path(area)) for area in ("temps", "state")):
        raise PipelineError("Cannot import a temporary or state file")
    digest, size = digest_file(source)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] or "video"
    suffix = source.suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) is None:
        raise PipelineError("Source needs a video filename extension")
    storage.initialize()
    destination = storage.path(f"artifacts/raw_footages/{stem}-{digest}{suffix}")
    with storage.run_lock(f"import-{digest}"):
        if destination.exists():
            if digest_file(destination) != (digest, size):
                raise ExecutionError(
                    "An existing raw footage import has changed; it was preserved"
                )
        else:
            stage = storage.path(f"artifacts/raw_footages/.{uuid.uuid4().hex}.partial")
            try:
                with source.open("rb") as original, stage.open("xb") as output:
                    if os.name != "nt":
                        os.fchmod(output.fileno(), 0o600)
                    shutil.copyfileobj(original, output, 1024 * 1024)
                if digest_file(stage) != (digest, size) or digest_file(source) != (
                    digest,
                    size,
                ):
                    raise ExecutionError("Video changed while importing")
                stage.rename(destination)
            finally:
                stage.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": digest, "size_bytes": size}


def input_records(inputs: dict, storage: UserStorage) -> dict:
    """Freeze input paths and hashes before any execution starts."""
    records = {}
    for name, value in inputs.items():
        path = Path(value).expanduser().resolve()
        if not path.is_file() or any(
            path.is_relative_to(storage.path(area)) for area in ("temps", "state")
        ):
            raise PipelineError(f"Input must be a permanent artifact file: {path}")
        digest, size = digest_file(path)
        records[name] = {"path": str(path), "sha256": digest, "size_bytes": size}
    return records


def create_workflow(args, storage: UserStorage, registry) -> dict:
    """Build the complete anonymization DAG with frozen registry/catalog inputs."""
    # Import only at call time to keep CLI definition helpers backward-compatible.
    from .cli import video_anonymization_definition
    from .face_data import bundled_yunet
    from .identity_data import bundled_sface

    if args.output.expanduser().exists() or args.output.expanduser().is_symlink():
        raise PipelineError("Job already exists; choose another --output")
    if args.interactive:
        args.video = Path(
            prompt("Ruta del video", str(args.video) if args.video else None)
        )
        args.device = prompt("Dispositivo de swap", args.device, ("cpu", "cuda"))
        args.codec = prompt("Codec final", args.codec, ("h264", "ffv1"))
    if args.video is None:
        raise PipelineError("Provide --video or use --interactive")
    definition = video_anonymization_definition(
        swap_parameters={
            "device": args.device,
            "chunk_frames": args.chunk_frames,
            "timeout_seconds": args.timeout_seconds,
        },
        export_parameters={
            "codec": args.codec,
            "crf": args.crf,
            "timeout_seconds": args.timeout_seconds,
            "allow_all_redacted": args.allow_all_redacted,
        },
    )
    Pipeline.parse(definition, registry)
    prepared = prepared_swap_paths(storage)
    inputs = {
        "video": args.video,
        "detector": bundled_yunet(),
        "recognizer": bundled_sface(),
        "swap_model": args.swap_model or prepared["swap_model"],
        "arcface_model": args.arcface_model or prepared["arcface_model"],
    }
    # Fail missing large assets before generating registry snapshots.
    records = input_records(inputs, storage)
    virtual = VirtualRegistry(storage)
    if not virtual.list_faces()["faces"]:
        raise PipelineError(
            "No approved B: register A, virtual create, then virtual approve --reviewed"
        )
    records.update(
        input_records(
            {
                "registry": IdentityRegistry(storage).snapshot(),
                "catalog": virtual.snapshot(),
            },
            storage,
        )
    )
    document = {
        "schema_version": 1,
        "storage_root": str(storage.root),
        "pipeline": definition,
        "inputs": records,
    }
    path = save_document(args.output, document)
    return {
        "job_path": str(path),
        "final": definition["final"],
        "nodes": [node["id"] for node in definition["nodes"]],
        "inputs": records,
        "next_argv": [
            "content-agent",
            "--root",
            str(storage.root),
            "workflow",
            "run",
            str(path),
        ],
    }


def load_workflow(path: Path, storage: UserStorage, registry):
    """Reject changed bindings and validate the DAG before calling Runner."""
    job = read_json(path)
    if (
        not isinstance(job, dict)
        or set(job) != {"schema_version", "storage_root", "pipeline", "inputs"}
        or type(job["schema_version"]) is not int
        or job["schema_version"] != 1
        or job["storage_root"] != str(storage.root)
        or not isinstance(job["inputs"], dict)
    ):
        raise PipelineError(
            "Invalid job or storage root differs; use the original --root / CONTENT_AGENT_HOME"
        )
    inputs = {}
    for name, record in job["inputs"].items():
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size_bytes"}
            or not isinstance(record["path"], str)
            or not Path(record["path"]).is_absolute()
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
        ):
            raise PipelineError("Job requires absolute frozen input bindings")
        inputs[name] = Path(record["path"])
    if input_records(inputs, storage) != job["inputs"]:
        raise PipelineError(
            "Job input changed; restore the original artifact or create a new job"
        )
    return Pipeline.parse(job["pipeline"], registry), inputs


def _run_row(storage: UserStorage, run_id: str):
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise PipelineError("Execution ID must contain 32 lowercase hexadecimal digits")
    storage.initialize()
    with storage.connect() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise PipelineError("Unknown execution")
    return dict(row)


def inspect_runs(storage: UserStorage, run_id=None, limit=20) -> dict:
    """Show execution/checkpoint status without reading biometric contents."""
    storage.initialize()
    if run_id is None:
        if not 1 <= limit <= 1000:
            raise PipelineError("Limit must be between 1 and 1000")
        with storage.connect() as connection:
            rows = connection.execute(
                "SELECT id,status,error,created_at,updated_at FROM runs ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"runs": [dict(row) for row in rows]}
    row = _run_row(storage, run_id)
    specification = json.loads(row.pop("specification"))
    with storage.connect() as connection:
        nodes = connection.execute(
            "SELECT node_id,status,error,outputs FROM nodes WHERE run_id=? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    return {
        **row,
        "inputs": specification["inputs"],
        "pipeline": specification["pipeline"],
        "nodes": [
            {
                **dict(node),
                "outputs": json.loads(node["outputs"]) if node["outputs"] else None,
            }
            for node in nodes
        ],
    }


def resume_run(storage: UserStorage, registry, run_id: str, cache="reuse", emit=None):
    """Resume with the original persisted paths, snapshots and parameters."""
    row = _run_row(storage, run_id)
    specification = json.loads(row["specification"])
    pipeline = Pipeline.parse(specification["pipeline"], registry)
    return Runner(storage, registry, emit=emit).run(
        pipeline,
        {name: Path(path) for name, path in specification["inputs"].items()},
        resume=run_id,
        cache=cache,
    )


def embedding_artifacts(storage: UserStorage, registry, run_id: str) -> dict:
    """Find and integrity-check a completed embedding node from any DAG."""
    row = _run_row(storage, run_id)
    definition = json.loads(row["specification"])["pipeline"]
    nodes = [
        node["id"] for node in definition["nodes"] if node["module"] == "face.embed"
    ]
    if len(nodes) != 1:
        raise PipelineError("Execution needs exactly one face.embed node")
    with storage.connect() as connection:
        found = connection.execute(
            "SELECT outputs FROM nodes WHERE run_id=? AND node_id=? AND status='succeeded'",
            (run_id, nodes[0]),
        ).fetchone()
    artifacts = (
        storage.verified_artifacts(found[0], dict(registry["face.embed"].output_kinds))
        if found
        else None
    )
    if artifacts is None:
        raise PipelineError("No intact completed embeddings in this execution")
    return artifacts


def track_summary(storage: UserStorage, registry, run_id: str) -> dict:
    """List explicitly selectable tracks with usable reference counts."""
    artifacts = embedding_artifacts(storage, registry, run_id)
    tracks = {}
    for row in iter_rows(artifacts["frames"].path):
        for face in row["faces"]:
            track = tracks.setdefault(
                face["track_id"],
                {
                    "track_id": face["track_id"],
                    "observations": 0,
                    "usable_observations": 0,
                },
            )
            track["observations"] += 1
            track["usable_observations"] += face["embedding_status"] == "usable"
    return {
        "run_id": run_id,
        "tracks": sorted(tracks.values(), key=lambda item: item["track_id"]),
        "review": "Select a track belonging only to the person you intend to register; at least 3 coherent usable observations are required",
    }


def result_payload(result) -> dict:
    """Keep saved-workflow and recovery output compatible with the existing CLI."""
    return {
        "run_id": result.run_id,
        "status": "succeeded",
        "cache_hits": list(result.cache_hits),
        "outputs": {
            name: {
                "path": str(item.path),
                "sha256": item.sha256,
                "size_bytes": item.size,
                "kind": item.kind,
            }
            for name, item in result.outputs.items()
        },
    }
