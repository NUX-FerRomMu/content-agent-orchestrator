"""CLI for inspecting inputs and executing declared pipelines."""

import argparse
import json
import os
import sys
from pathlib import Path

from .contracts import ExecutionError, PipelineError
from .face_data import bundled_yunet
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
    run = commands.add_parser("run", help="Run a JSON pipeline")
    run.add_argument("pipeline", type=Path)
    run.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    for command in (inspect, analyze, run):
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
        print(
            json.dumps(
                {
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
    except (PipelineError, ExecutionError, OSError, ValueError) as exc:
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
