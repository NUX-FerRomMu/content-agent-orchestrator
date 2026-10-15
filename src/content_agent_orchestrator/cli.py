"""CLI for inspecting inputs and executing declared pipelines."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .face_data import bundled_yunet, read_json
from .generation_models import MODEL_REPOSITORIES, prepare_model
from .identity_data import bundled_sface, validate_snapshot
from .identity_registry import IdentityRegistry
from .modules import builtin_registry
from .pipeline import Pipeline
from .portrait_generators import CommandPortraitGenerator
from .runner import Runner
from .sdxl_generator import SDXLPortraitGenerator
from .storage import UserStorage
from .virtual_registry import VirtualRegistry


def inspection_definition() -> dict:
    return {
        "nodes": [
            {
                "id": "inspect",
                "module": "video.inspect",
                "inputs": {"video": {"external": "video"}},
            },
            {
                "id": "export",
                "module": "report.export",
                "inputs": {"inspection": {"node": "inspect", "output": "manifest"}},
            },
        ],
        "final": "export",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local composable video pipelines: register A, approve synthetic B, render validated video.",
        epilog="Start: content-agent init; content-agent doctor; content-agent workflow create --help. Full guide: docs/quickstart-local.md",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get(
                "CONTENT_AGENT_HOME", str(Path.home() / ".content-agent-orchestrator")
            )
        ),
        help="User storage directory (or CONTENT_AGENT_HOME)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _workflow_parsers(commands)
    inspect = commands.add_parser(
        "inspect", help="Inspect a video and export its report"
    )
    inspect.add_argument("video", type=Path)
    analyze = commands.add_parser(
        "analyze-faces",
        help="Detect and track faces; export metadata, not an anonymized video",
    )
    analyze.add_argument("video", type=Path)
    analyze.add_argument(
        "--model",
        type=Path,
        default=bundled_yunet(),
        help="Pinned YuNet model (bundled by default)",
    )
    analyze.add_argument(
        "--max-dimension",
        type=int,
        default=640,
        help="Maximum detection image dimension, a multiple of 32",
    )
    analyze.add_argument("--score-threshold", type=float, default=0.85)
    recognize = commands.add_parser(
        "recognize-faces",
        help="Extract private embeddings and resolve identities; does not anonymize",
    )
    recognize.add_argument("video", type=Path)
    recognize.add_argument("--detector", type=Path, default=bundled_yunet())
    recognize.add_argument("--recognizer", type=Path, default=bundled_sface())
    recognize.add_argument(
        "--snapshot",
        type=Path,
        help="Explicit frozen registry snapshot; the original is required with --resume",
    )
    recognize.add_argument("--min-similarity", type=float, default=0.6)
    recognize.add_argument("--ambiguity-margin", type=float, default=0.12)
    virtualize = commands.add_parser(
        "assign-virtual-faces",
        help="Map known faces to approved synthetic targets; does not swap video",
    )
    virtualize.add_argument("video", type=Path)
    virtualize.add_argument("--detector", type=Path, default=bundled_yunet())
    virtualize.add_argument("--recognizer", type=Path, default=bundled_sface())
    virtualize.add_argument(
        "--snapshot",
        type=Path,
        help="Original registry snapshot required with --resume",
    )
    virtualize.add_argument(
        "--catalog", type=Path, help="Original virtual catalog required with --resume"
    )
    virtualize.add_argument("--min-similarity", type=float, default=0.6)
    virtualize.add_argument("--ambiguity-margin", type=float, default=0.12)
    anonymize = commands.add_parser(
        "anonymize-video",
        help="Replace approved faces and publish only validated/redacted video",
    )
    anonymize.add_argument("video", type=Path)
    anonymize.add_argument("--detector", type=Path, default=bundled_yunet())
    anonymize.add_argument("--recognizer", type=Path, default=bundled_sface())
    anonymize.add_argument(
        "--snapshot",
        type=Path,
        help="Original registry snapshot required with --resume",
    )
    anonymize.add_argument(
        "--catalog", type=Path, help="Original virtual catalog required with --resume"
    )
    anonymize.add_argument("--swap-model", type=Path, required=True)
    anonymize.add_argument("--arcface-model", type=Path, required=True)
    anonymize.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    anonymize.add_argument("--chunk-frames", type=int, default=32)
    anonymize.add_argument("--timeout-seconds", type=int, default=600)
    anonymize.add_argument("--codec", choices=("h264", "ffv1"), default="h264")
    anonymize.add_argument("--crf", type=int, default=18)
    anonymize.add_argument("--allow-all-redacted", action="store_true")
    anonymize.add_argument("--min-similarity", type=float, default=0.6)
    anonymize.add_argument("--ambiguity-margin", type=float, default=0.12)
    virtual = commands.add_parser(
        "virtual", help="Generate, visually approve and reuse immutable synthetic faces"
    )
    virtual_actions = virtual.add_subparsers(dest="virtual_action", required=True)
    create = virtual_actions.add_parser("create")
    create.add_argument("identity_id")
    create.add_argument("--profile", type=Path, required=True)
    create.add_argument("--backend", choices=("sdxl", "command"), default="sdxl")
    create.add_argument(
        "--model-dir", type=Path, help="Directory returned by virtual prepare-model"
    )
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--steps", type=int, default=30)
    create.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    create.add_argument(
        "--offload", action=argparse.BooleanOptionalAction, default=True
    )
    create.add_argument("--cache", choices=("reuse", "fresh"), default="reuse")
    create.add_argument(
        "--generator", type=Path, help="Local command configuration JSON"
    )
    approve = virtual_actions.add_parser("approve")
    approve.add_argument("identity_id")
    approve.add_argument("candidate_id")
    approve.add_argument(
        "--reviewed",
        action="store_true",
        help="I reviewed B: fictional generation, clearly adult, no eyewear/eccentric traits, gender/skin match A",
    )
    virtual_actions.add_parser("list")
    virtual_actions.add_parser("snapshot")
    prepare = virtual_actions.add_parser(
        "prepare-model",
        help="Explicit download and checksum verification of open SDXL weights",
    )
    prepare.add_argument(
        "--model", choices=tuple(MODEL_REPOSITORIES), default="realvis-xl"
    )
    prepare.add_argument("--revision", default="main")
    identity = commands.add_parser(
        "identity", help="Explicit local identity enrollment and management"
    )
    actions = identity.add_subparsers(dest="identity_action", required=True)
    register = actions.add_parser("register")
    register.add_argument("--embeddings", type=Path)
    register.add_argument("--summary", type=Path)
    register.add_argument(
        "--from-run", help="Read verified private embeddings from an execution ID"
    )
    register.add_argument("--track", required=True)
    target = register.add_mutually_exclusive_group()
    target.add_argument("--alias")
    target.add_argument(
        "--identity-id", help="Add coherent references to an existing identity"
    )
    register.add_argument("--min-similarity", type=float, default=0.65)
    actions.add_parser("list")
    tracks = actions.add_parser(
        "tracks", help="List tracks and usable observations for explicit enrollment"
    )
    tracks.add_argument("--run", required=True)
    actions.add_parser("snapshot")
    delete = actions.add_parser("delete")
    delete.add_argument("identity_id")
    run = commands.add_parser("run", help="Run a JSON pipeline")
    run.add_argument("pipeline", type=Path)
    run.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    for command in (inspect, analyze, recognize, virtualize, anonymize, run):
        command.add_argument("--cache", choices=("reuse", "fresh"), default="reuse")
        command.add_argument(
            "--resume",
            metavar="RUN_ID",
            help="Resume a failed/interrupted run with identical configuration",
        )
    args = parser.parse_args(argv)
    try:
        registry = builtin_registry()
        storage = UserStorage(args.root)
        if args.command in {
            "init",
            "doctor",
            "models",
            "artifacts",
            "profile",
            "workflow",
            "runs",
            "modules",
        }:
            payload = _workflow_command(args, storage, registry)
            print(json.dumps(payload, indent=2))
            return 1 if args.command == "doctor" and not payload["ready"] else 0
        if args.command == "virtual" and args.virtual_action == "prepare-model":
            path = prepare_model(storage, args.model, args.revision)
            print(
                json.dumps(
                    {"model_dir": str(path), "generation_is_offline": True}, indent=2
                )
            )
            return 0
        if args.command == "virtual":
            faces = VirtualRegistry(storage)
            if args.virtual_action == "create":
                if args.backend == "command":
                    if args.generator is None:
                        raise PipelineError(
                            "Local generation requires --generator configuration"
                        )
                    generator = CommandPortraitGenerator(read_json(args.generator))
                else:
                    if args.generator is not None:
                        raise PipelineError(
                            "--generator is only for the command backend"
                        )
                    model_dir = args.model_dir or storage.path(
                        "artifacts/models/unprepared"
                    )
                    generator = SDXLPortraitGenerator(
                        model_dir,
                        seed=args.seed,
                        steps=args.steps,
                        device=args.device,
                        offload=args.offload,
                    )
                payload = faces.create(
                    args.identity_id,
                    read_json(args.profile),
                    generator,
                    cache=args.cache,
                )
            elif args.virtual_action == "approve":
                payload = faces.approve(
                    args.identity_id, args.candidate_id, reviewed=args.reviewed
                )
            elif args.virtual_action == "list":
                payload = faces.list_faces()
            else:
                payload = {"path": str(faces.snapshot())}
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "identity":
            identities = IdentityRegistry(storage)
            if args.identity_action == "register":
                if args.from_run:
                    if args.embeddings is not None or args.summary is not None:
                        raise PipelineError(
                            "Use --from-run or both --embeddings and --summary"
                        )
                    from .workflow import embedding_artifacts

                    artifacts = embedding_artifacts(storage, registry, args.from_run)
                    args.embeddings, args.summary = (
                        artifacts["frames"].path,
                        artifacts["summary"].path,
                    )
                elif args.embeddings is None or args.summary is None:
                    raise PipelineError(
                        "Provide --from-run or both --embeddings and --summary"
                    )
                payload = identities.enroll(
                    args.embeddings,
                    args.summary,
                    args.track,
                    alias=args.alias,
                    identity_id=args.identity_id,
                    min_similarity=args.min_similarity,
                )
            elif args.identity_action == "list":
                payload = identities.list_identities()
            elif args.identity_action == "tracks":
                from .workflow import track_summary

                payload = track_summary(storage, registry, args.run)
            elif args.identity_action == "snapshot":
                payload = {"path": str(identities.snapshot())}
            else:
                payload = identities.delete(args.identity_id)
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "inspect":
            pipeline = Pipeline.parse(inspection_definition(), registry)
            inputs = {"video": args.video}
        elif args.command == "analyze-faces":
            pipeline = Pipeline.parse(
                face_analysis_definition(
                    {
                        "max_dimension": args.max_dimension,
                        "score_threshold": args.score_threshold,
                    }
                ),
                registry,
            )
            inputs = {"video": args.video, "detector": args.model}
        elif args.command in (
            "recognize-faces",
            "assign-virtual-faces",
            "anonymize-video",
        ):
            if args.resume and args.snapshot is None:
                raise PipelineError("Resuming requires the original --snapshot")
            if (
                args.resume
                and args.command in ("assign-virtual-faces", "anonymize-video")
                and args.catalog is None
            ):
                raise PipelineError("Resuming requires the original --catalog")
            snapshot = args.snapshot or IdentityRegistry(storage).snapshot()
            validate_snapshot(read_json(snapshot))
            definition = (
                virtual_assignment_definition
                if args.command == "assign-virtual-faces"
                else identity_recognition_definition
            )
            extra_parameters = {}
            if args.command == "anonymize-video":
                definition = video_anonymization_definition
                extra_parameters = {
                    "swap_parameters": {
                        "device": args.device,
                        "chunk_frames": args.chunk_frames,
                        "timeout_seconds": args.timeout_seconds,
                    },
                    "export_parameters": {
                        "codec": args.codec,
                        "crf": args.crf,
                        "timeout_seconds": args.timeout_seconds,
                        "allow_all_redacted": args.allow_all_redacted,
                    },
                }
            pipeline = Pipeline.parse(
                definition(
                    **extra_parameters,
                    resolution_parameters={
                        "min_similarity": args.min_similarity,
                        "ambiguity_margin": args.ambiguity_margin,
                    },
                ),
                registry,
            )
            inputs = {
                "video": args.video,
                "detector": args.detector,
                "recognizer": args.recognizer,
                "registry": snapshot,
            }
            if args.command in ("assign-virtual-faces", "anonymize-video"):
                inputs["catalog"] = args.catalog or VirtualRegistry(storage).snapshot()
            if args.command == "anonymize-video":
                inputs["swap_model"] = args.swap_model
                inputs["arcface_model"] = args.arcface_model
        else:
            pipeline = Pipeline.load(args.pipeline, registry)
            inputs = {}
            for binding in args.input:
                name, separator, value = binding.partition("=")
                if not separator or not name or not value or name in inputs:
                    raise PipelineError("Inputs must be unique NAME=PATH bindings")
                inputs[name] = Path(value)
        result = Runner(
            storage,
            registry,
            emit=lambda event: print(json.dumps(event), file=sys.stderr),
        ).run(pipeline, inputs, cache=args.cache, resume=args.resume)
        extra = {}
        if args.command in (
            "recognize-faces",
            "assign-virtual-faces",
            "anonymize-video",
        ):
            with storage.connect() as connection:
                encoded = connection.execute(
                    "SELECT outputs FROM nodes WHERE run_id = ? AND node_id = 'embed' AND status = 'succeeded'",
                    (result.run_id,),
                ).fetchone()
            artifacts = (
                storage.verified_artifacts(
                    encoded[0], dict(registry["face.embed"].output_kinds)
                )
                if encoded
                else None
            )
            if artifacts is None:
                raise ExecutionError("Cannot verify completed embedding artifacts")
            extra = {
                "registry_snapshot": str(Path(inputs["registry"]).resolve()),
                "private_intermediates": {
                    name: str(item.path) for name, item in artifacts.items()
                },
            }
            if args.command in ("assign-virtual-faces", "anonymize-video"):
                extra["virtual_catalog"] = str(Path(inputs["catalog"]).resolve())
        print(
            json.dumps(
                {
                    **extra,
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
                },
                indent=2,
            )
        )
        return 0
    except (PipelineError, ExecutionError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; temporaries and checkpoints are retained", file=sys.stderr)
        return 130


def _workflow_parsers(commands):
    """Describe preparation, editable saved jobs and checkpoint recovery."""
    commands.add_parser(
        "init", help="Create private artifacts, temps, outputs and state directories"
    )
    doctor = commands.add_parser(
        "doctor",
        help="Check tools, optional runtimes and model checksums without downloading",
    )
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    doctor.add_argument(
        "--generation",
        action="store_true",
        help="Also check the optional local SDXL runtime",
    )
    doctor.add_argument(
        "--models", action="store_true", help="Also verify prepared native swap weights"
    )
    models = commands.add_parser("models", help="Explicit verified model preparation")
    models.add_subparsers(dest="models_action", required=True).add_parser(
        "prepare-swap",
        help="Download pinned InSwapper128 + ArcFace weights (~696 MiB); keep completed files on retry",
    )
    artifacts = commands.add_parser("artifacts", help="Manage initial user artifacts")
    importer = artifacts.add_subparsers(
        dest="artifacts_action", required=True
    ).add_parser(
        "import",
        help="Copy a video to raw_footages with a content hash; preserve the original",
    )
    importer.add_argument("video", type=Path)
    profile = commands.add_parser(
        "profile", help="Create the explicit adult synthetic appearance profile"
    )
    creator = profile.add_subparsers(dest="profile_action", required=True).add_parser(
        "create"
    )
    creator.add_argument("--output", type=Path, required=True)
    creator.add_argument("--interactive", action="store_true")
    creator.add_argument("--gender", choices=("woman", "man", "androgynous"))
    creator.add_argument(
        "--skin-tone", type=int, help="Explicit Monk skin tone 1 (light) to 10 (dark)"
    )
    creator.add_argument(
        "--undertone", choices=("neutral", "warm", "cool"), default="neutral"
    )
    creator.add_argument("--age-years", type=int, default=35)
    creator.add_argument(
        "--features",
        default="",
        help="Cosmetic preferences, e.g. natural makeup and subtle eyeliner",
    )
    workflow = commands.add_parser(
        "workflow", help="Build and run a saved JSON job with a DAG and frozen inputs"
    )
    workflows = workflow.add_subparsers(dest="workflow_action", required=True)
    builder = workflows.add_parser(
        "create",
        help="Assemble the full video pipeline; requires registered A and approved B",
    )
    builder.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON job; existing files are never overwritten",
    )
    builder.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for video, CPU/CUDA and final codec",
    )
    builder.add_argument("--video", type=Path)
    builder.add_argument(
        "--swap-model",
        type=Path,
        help="Default: pinned weights from models prepare-swap",
    )
    builder.add_argument("--arcface-model", type=Path)
    builder.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    builder.add_argument("--codec", choices=("h264", "ffv1"), default="h264")
    builder.add_argument("--crf", type=int, default=18)
    builder.add_argument("--chunk-frames", type=int, default=32)
    builder.add_argument("--timeout-seconds", type=int, default=600)
    builder.add_argument("--allow-all-redacted", action="store_true")
    executor = workflows.add_parser(
        "run", help="Execute the saved job after verifying all frozen input hashes"
    )
    executor.add_argument("job", type=Path)
    executor.add_argument("--cache", choices=("reuse", "fresh"), default="reuse")
    runs = commands.add_parser(
        "runs",
        help="Discover execution IDs, outputs, failures and recoverable checkpoints",
    )
    run_actions = runs.add_subparsers(dest="runs_action", required=True)
    listing = run_actions.add_parser("list")
    listing.add_argument("--limit", type=int, default=20)
    show = run_actions.add_parser("show")
    show.add_argument("run_id")
    resume = run_actions.add_parser(
        "resume",
        help="Reuse the original recorded DAG, paths and parameters; no manual snapshots needed",
    )
    resume.add_argument("run_id")
    resume.add_argument("--cache", choices=("reuse", "fresh"), default="reuse")
    commands.add_parser(
        "modules",
        help="List available module contracts for custom JSON DAG composition",
    )


def _workflow_command(args, storage, registry):
    """Dispatch optional workflow helpers without coupling the runner to the CLI."""
    from .workflow import (
        create_profile,
        create_workflow,
        import_video,
        inspect_runs,
        load_workflow,
        result_payload,
        resume_run,
    )

    emit = lambda event: print(json.dumps(event), file=sys.stderr)
    if args.command == "init":
        storage.initialize()
        return {
            "storage_root": str(storage.root),
            "directories": [
                str(storage.path(name))
                for name in ("artifacts/raw_footages", "temps", "outputs", "state")
            ],
        }
    if args.command == "doctor":
        from .diagnostics import diagnose

        return diagnose(storage, args.device, args.generation, args.models)
    if args.command == "models":
        from .swap_models import prepare_swap_models

        return {name: str(path) for name, path in prepare_swap_models(storage).items()}
    if args.command == "artifacts":
        return import_video(storage, args.video)
    if args.command == "profile":
        return create_profile(args)
    if args.command == "modules":
        return {
            "modules": [
                {
                    "name": module.name,
                    "version": module.version,
                    "inputs": dict(module.input_kinds),
                    "outputs": dict(module.output_kinds),
                    "can_be_final": getattr(module, "can_be_final", True),
                }
                for module in registry.values()
            ]
        }
    if args.command == "workflow":
        if args.workflow_action == "create":
            return create_workflow(args, storage, registry)
        pipeline, inputs = load_workflow(args.job, storage, registry)
        return result_payload(
            Runner(storage, registry, emit=emit).run(pipeline, inputs, cache=args.cache)
        )
    if args.runs_action == "resume":
        return result_payload(
            resume_run(storage, registry, args.run_id, cache=args.cache, emit=emit)
        )
    return inspect_runs(
        storage, getattr(args, "run_id", None), getattr(args, "limit", 20)
    )


def face_analysis_definition(
    detection_parameters: dict | None = None, tracking_parameters: dict | None = None
) -> dict:
    return {
        "nodes": [
            {
                "id": "inspect",
                "module": "video.inspect",
                "inputs": {"video": {"external": "video"}},
            },
            {
                "id": "detect",
                "module": "face.detect",
                "inputs": {
                    "video": {"external": "video"},
                    "model": {"external": "detector"},
                    "inspection": {"node": "inspect", "output": "manifest"},
                    "timestamps": {"node": "inspect", "output": "timestamps"},
                },
                "parameters": detection_parameters or {},
            },
            {
                "id": "track",
                "module": "face.track",
                "inputs": {
                    "frames": {"node": "detect", "output": "frames"},
                    "summary": {"node": "detect", "output": "summary"},
                },
                "parameters": tracking_parameters or {},
            },
            {
                "id": "report",
                "module": "face.report",
                "inputs": {
                    name: {"node": "track", "output": name}
                    for name in ("summary", "frames", "tracks", "intervals")
                },
            },
        ],
        "final": "report",
    }


def identity_recognition_definition(
    detection_parameters=None,
    tracking_parameters=None,
    embedding_parameters=None,
    resolution_parameters=None,
) -> dict:
    definition = face_analysis_definition(detection_parameters, tracking_parameters)
    definition["nodes"].pop()
    definition["nodes"].extend(
        [
            {
                "id": "embed",
                "module": "face.embed",
                "parameters": embedding_parameters or {},
                "inputs": {
                    "video": {"external": "video"},
                    "model": {"external": "recognizer"},
                    "inspection": {"node": "inspect", "output": "manifest"},
                    "timestamps": {"node": "inspect", "output": "timestamps"},
                    "frames": {"node": "track", "output": "frames"},
                    "summary": {"node": "track", "output": "summary"},
                },
            },
            {
                "id": "resolve",
                "module": "identity.resolve",
                "parameters": resolution_parameters or {},
                "inputs": {
                    "frames": {"node": "embed", "output": "frames"},
                    "summary": {"node": "embed", "output": "summary"},
                    "registry": {"external": "registry"},
                },
            },
            {
                "id": "report",
                "module": "identity.report",
                "inputs": {
                    name: {"node": "resolve", "output": name}
                    for name in ("summary", "frames", "tracks")
                },
            },
        ]
    )
    return definition


def virtual_assignment_definition(
    detection_parameters=None,
    tracking_parameters=None,
    embedding_parameters=None,
    resolution_parameters=None,
) -> dict:
    definition = identity_recognition_definition(
        detection_parameters,
        tracking_parameters,
        embedding_parameters,
        resolution_parameters,
    )
    definition["nodes"].pop()
    definition["nodes"].extend(
        [
            {
                "id": "virtual",
                "module": "virtual.resolve",
                "inputs": {
                    "frames": {"node": "resolve", "output": "frames"},
                    "summary": {"node": "resolve", "output": "summary"},
                    "catalog": {"external": "catalog"},
                },
            },
            {
                "id": "report",
                "module": "virtual.report",
                "inputs": {
                    name: {"node": "virtual", "output": name}
                    for name in ("frames", "targets", "summary")
                },
            },
        ]
    )
    return definition


def video_anonymization_definition(
    *, swap_parameters=None, export_parameters=None, resolution_parameters=None
) -> dict:
    swap_parameters = swap_parameters or {}
    timeout = swap_parameters.get("timeout_seconds", 600)
    definition = virtual_assignment_definition(
        detection_parameters={"timeout_seconds": timeout},
        embedding_parameters={"timeout_seconds": timeout},
        resolution_parameters=resolution_parameters,
    )
    definition["nodes"].pop()
    definition["nodes"][0]["parameters"] = {"timeout_seconds": timeout}

    def node(name, output):
        return {"node": name, "output": output}

    def external(name):
        return {"external": name}

    shared = {"timestamps": node("inspect", "timestamps")}
    references = {
        "assignments": node("virtual", "frames"),
        "embeddings": node("embed", "frames"),
        "targets": node("virtual", "targets"),
        "registry": external("registry"),
        "detector": external("detector"),
        "recognizer": external("recognizer"),
    }
    swapped = {key: node("swap", key) for key in ("frames", "records", "summary")}
    definition["nodes"].extend(
        [
            {
                "id": "swap",
                "module": "face.swap",
                "parameters": swap_parameters,
                "inputs": shared
                | {
                    "video": external("video"),
                    "inspection": node("inspect", "manifest"),
                    "assignments": node("virtual", "frames"),
                    "summary": node("virtual", "summary"),
                    "targets": node("virtual", "targets"),
                    "swap_model": external("swap_model"),
                    "arcface_model": external("arcface_model"),
                },
            },
            {
                "id": "validate",
                "module": "face.validate",
                "inputs": shared | references | swapped,
            },
            {
                "id": "redact",
                "module": "face.redact",
                "parameters": {"chunk_frames": swap_parameters.get("chunk_frames", 32)},
                "inputs": shared
                | swapped
                | {
                    "audit": node("validate", "audit"),
                    "validation": node("validate", "summary"),
                },
            },
            {
                "id": "export",
                "module": "video.export",
                "parameters": export_parameters or {},
                "inputs": shared
                | references
                | {
                    "video": external("video"),
                    "inspection": node("inspect", "manifest"),
                    "swap_records": node("swap", "records"),
                    **{
                        key: node("redact", key)
                        for key in ("frames", "records", "summary")
                    },
                },
            },
        ]
    )
    definition["final"] = "export"
    return definition
