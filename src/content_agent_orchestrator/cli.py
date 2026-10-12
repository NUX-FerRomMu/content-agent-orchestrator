"""CLI for inspecting inputs and executing declared pipelines."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .face_data import bundled_yunet, read_json
from .identity_data import bundled_sface, validate_snapshot
from .identity_registry import IdentityRegistry
from .modules import builtin_registry
from .pipeline import Pipeline
from .runner import Runner
from .storage import UserStorage


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
    parser = argparse.ArgumentParser(description="Composable content pipelines")
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
        help="Explicit frozen registry snapshot, required to resume after registry changes",
    )
    recognize.add_argument("--min-similarity", type=float, default=0.6)
    recognize.add_argument("--ambiguity-margin", type=float, default=0.12)
    identity = commands.add_parser(
        "identity", help="Explicit local identity enrollment and management"
    )
    actions = identity.add_subparsers(dest="identity_action", required=True)
    register = actions.add_parser("register")
    register.add_argument("--embeddings", type=Path, required=True)
    register.add_argument("--summary", type=Path, required=True)
    register.add_argument("--track", required=True)
    target = register.add_mutually_exclusive_group()
    target.add_argument("--alias")
    target.add_argument(
        "--identity-id", help="Add coherent references to an existing identity"
    )
    register.add_argument("--min-similarity", type=float, default=0.65)
    actions.add_parser("list")
    actions.add_parser("snapshot")
    delete = actions.add_parser("delete")
    delete.add_argument("identity_id")
    run = commands.add_parser("run", help="Run a JSON pipeline")
    run.add_argument("pipeline", type=Path)
    run.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    for command in (inspect, analyze, recognize, run):
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
        if args.command == "identity":
            identities = IdentityRegistry(storage)
            if args.identity_action == "register":
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
        elif args.command == "recognize-faces":
            snapshot = args.snapshot or IdentityRegistry(storage).snapshot()
            validate_snapshot(read_json(snapshot))
            pipeline = Pipeline.parse(
                identity_recognition_definition(
                    resolution_parameters={
                        "min_similarity": args.min_similarity,
                        "ambiguity_margin": args.ambiguity_margin,
                    }
                ),
                registry,
            )
            inputs = {
                "video": args.video,
                "detector": args.detector,
                "recognizer": args.recognizer,
                "registry": snapshot,
            }
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
        if args.command == "recognize-faces":
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
