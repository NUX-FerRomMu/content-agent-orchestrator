"""Sequential execution with content-verified reuse and atomic publication."""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Artifact, ExecutionError, Module, ModuleContext, PipelineError
from .pipeline import Pipeline, canonical_json, valid_identifier
from .storage import UserStorage, digest_file


class RunFailed(ExecutionError):
    def __init__(self, run_id: str, message: str):
        self.run_id = run_id
        super().__init__(f"Run {run_id} failed: {message}")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outputs: Mapping[str, Artifact]
    cache_hits: tuple[str, ...]


class Runner:
    def __init__(
        self,
        storage: UserStorage,
        registry: Mapping[str, Module],
        emit: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.storage = storage
        self.registry = registry
        self.emit = emit or (lambda event: None)

    def run(
        self,
        pipeline: Pipeline,
        inputs: Mapping[str, Path],
        *,
        cache: str = "reuse",
        resume: str | None = None,
    ) -> RunResult:
        if cache not in {"reuse", "fresh"}:
            raise PipelineError("Cache mode must be 'reuse' or 'fresh'")
        # Revalidate in case a caller constructed or modified a Pipeline directly.
        pipeline = Pipeline.parse(pipeline.definition, self.registry)
        expected = {}
        for node in pipeline.nodes:
            for name, reference in node.inputs.items():
                if reference.external:
                    expected[reference.external] = self.registry[
                        node.module
                    ].input_kinds[name]
        if set(inputs) != set(expected):
            raise PipelineError(
                f"Required external inputs: {', '.join(sorted(expected))}"
            )
        external = {}
        for name, supplied in inputs.items():
            path = supplied.expanduser().resolve()
            if not path.is_file():
                raise PipelineError(f"Input is not a file: {path}")
            for area in ("temps", "state"):
                if path.is_relative_to(self.storage.path(area).resolve()):
                    raise PipelineError(
                        "Inputs cannot come from temporary or state storage"
                    )
            digest, size = digest_file(path)
            external[name] = Artifact(path, expected[name], digest, size)
        specification = canonical_json(
            {
                "pipeline": pipeline.definition,
                "inputs": {name: str(item.path) for name, item in external.items()},
                "versions": {
                    node.module: self.registry[node.module].version
                    for node in pipeline.nodes
                },
            }
        )
        run_id = resume if resume is not None else uuid.uuid4().hex
        if (
            not isinstance(run_id, str)
            or len(run_id) != 32
            or any(char not in "0123456789abcdef" for char in run_id)
        ):
            raise PipelineError(
                "Execution ID must contain 32 lowercase hexadecimal digits"
            )
        self.storage.initialize()
        with self.storage.run_lock(run_id):
            with self.storage.connect() as connection:
                previous = connection.execute(
                    "SELECT * FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if resume is not None:
                    if previous is None:
                        raise PipelineError("Cannot resume an unknown execution")
                    if previous["specification"] != specification:
                        raise PipelineError(
                            "Resume requires the same pipeline, paths and versions"
                        )
                    if previous["status"] == "succeeded":
                        raise PipelineError(
                            "Execution already completed; start a new run"
                        )
                    connection.execute(
                        "UPDATE runs SET status='running', error=NULL, "
                        "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (run_id,),
                    )
                else:
                    connection.execute(
                        "INSERT INTO runs(id, specification, status) VALUES (?, ?, 'running')",
                        (run_id, specification),
                    )
            self.emit({"event": "run_started", "run_id": run_id})
            results: dict[str, dict[str, Artifact]] = {}
            hits = []
            try:
                for node in pipeline.nodes:
                    module = self.registry[node.module]
                    bound = {
                        name: external[reference.external]
                        if reference.external
                        else results[reference.node][reference.output]
                        for name, reference in node.inputs.items()
                    }
                    # Catch edits during an execution, including upstream artifacts.
                    self._verify_inputs(bound)
                    fingerprint = hashlib.sha256(
                        canonical_json(
                            {
                                "module": module.name,
                                "version": module.version,
                                "parameters": node.parameters,
                                "inputs": {
                                    name: {
                                        "kind": item.kind,
                                        "sha256": item.sha256,
                                        "size": item.size,
                                    }
                                    for name, item in bound.items()
                                },
                            }
                        ).encode()
                    ).hexdigest()
                    cached = (
                        self._cached(fingerprint, dict(module.output_kinds))
                        if cache == "reuse"
                        else None
                    )
                    with self.storage.connect() as connection:
                        previous_node = connection.execute(
                            "SELECT fingerprint FROM nodes WHERE run_id=? AND node_id=?",
                            (run_id, node.id),
                        ).fetchone()
                    stale_work = (
                        previous_node is not None
                        and previous_node["fingerprint"] != fingerprint
                    )
                    work = self.storage.path(f"temps/{run_id}/{node.id}")
                    if work.is_symlink():
                        raise PipelineError("Module workspace must not be a symlink")
                    if (cache == "fresh" or stale_work) and work.exists():
                        shutil.rmtree(work)
                    # Record the new fingerprint only after incompatible work is
                    # gone, so a kill between these steps cannot bless stale cache.
                    self._node_checkpoint(run_id, node.id, fingerprint, "running")
                    work.mkdir(parents=True, exist_ok=True, mode=0o700)
                    result_dir = work / "result"
                    if result_dir.is_symlink():
                        raise PipelineError(
                            "Module result directory must not be a symlink"
                        )
                    # Partial outputs are never interpreted as a successful cache.
                    if result_dir.exists():
                        shutil.rmtree(result_dir)
                    result_dir.mkdir(mode=0o700)
                    self.emit(
                        {"event": "node_started", "run_id": run_id, "node_id": node.id}
                    )
                    if cached is not None:
                        sources = cached
                        hits.append(node.id)
                    else:
                        paths = module.execute(
                            bound, node.parameters, ModuleContext(work, result_dir)
                        )
                        sources = self._collect(paths, module, result_dir)
                    self._verify_inputs(bound)
                    published, created = self._publish(
                        run_id,
                        node.id,
                        fingerprint,
                        sources,
                        work,
                        node.id == pipeline.final,
                    )
                    try:
                        self._node_checkpoint(
                            run_id, node.id, fingerprint, "succeeded", published
                        )
                        if node.id == pipeline.final:
                            with self.storage.connect() as connection:
                                connection.execute(
                                    "UPDATE runs SET status='succeeded', error=NULL, "
                                    "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                    (run_id,),
                                )
                    except BaseException:
                        # Roll back only a publication created by this attempt.
                        if created:
                            directory = next(iter(published.values())).path.parent
                            shutil.rmtree(directory)
                        raise
                    results[node.id] = published
                    try:
                        shutil.rmtree(work)
                    except OSError:
                        self.emit({"event": "cleanup_warning", "node_id": node.id})
                    self.emit(
                        {
                            "event": "node_succeeded",
                            "node_id": node.id,
                            "cached": cached is not None,
                        }
                    )
                return RunResult(run_id, results[pipeline.final], tuple(hits))
            except BaseException as exc:
                with self.storage.connect() as connection:
                    connection.execute(
                        "UPDATE nodes SET status='failed', error=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE run_id=? AND status='running'",
                        (str(exc)[:1000], run_id),
                    )
                    connection.execute(
                        "UPDATE runs SET status='failed', error=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (str(exc)[:1000], run_id),
                    )
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    self.emit({"event": "run_interrupted", "run_id": run_id})
                    raise
                raise RunFailed(run_id, str(exc)) from exc

    @staticmethod
    def _verify_inputs(inputs: Mapping[str, Artifact]) -> None:
        for item in inputs.values():
            if digest_file(item.path) != (item.sha256, item.size):
                raise ExecutionError("An input changed during execution")

    def _cached(
        self, fingerprint: str, kinds: dict[str, str]
    ) -> dict[str, Artifact] | None:
        with self.storage.connect() as connection:
            records = connection.execute(
                "SELECT outputs FROM nodes WHERE fingerprint=? AND status='succeeded' "
                "ORDER BY updated_at DESC",
                (fingerprint,),
            ).fetchall()
        for record in records:
            verified = self.storage.verified_artifacts(record["outputs"], kinds)
            if verified is not None:
                return verified
        return None

    def _node_checkpoint(
        self,
        run_id: str,
        node_id: str,
        fingerprint: str,
        status: str,
        outputs: dict[str, Artifact] | None = None,
    ) -> None:
        with self.storage.connect() as connection:
            connection.execute(
                "INSERT INTO nodes(run_id,node_id,fingerprint,status,outputs) VALUES(?,?,?,?,?) "
                "ON CONFLICT(run_id,node_id) DO UPDATE SET fingerprint=excluded.fingerprint, "
                "status=excluded.status,outputs=excluded.outputs,error=NULL, "
                "updated_at=CURRENT_TIMESTAMP",
                (
                    run_id,
                    node_id,
                    fingerprint,
                    status,
                    self.storage.serialize_artifacts(outputs)
                    if outputs is not None
                    else None,
                ),
            )

    @staticmethod
    def _collect(
        paths: Mapping[str, Path], module: Module, result_dir: Path
    ) -> dict[str, Artifact]:
        if not module.output_kinds or set(paths) != set(module.output_kinds):
            raise ExecutionError("Module did not return its declared outputs")
        artifacts = {}
        seen = set()
        for name, raw_path in paths.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = result_dir / path
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(result_dir.resolve())
                or not path.is_file()
                or any(
                    parent.is_symlink()
                    for parent in path.parents
                    if parent != result_dir
                )
                or path.resolve() in seen
            ):
                raise ExecutionError(
                    "Outputs must be distinct regular files inside result_dir"
                )
            seen.add(path.resolve())
            digest, size = digest_file(path)
            artifacts[name] = Artifact(path, module.output_kinds[name], digest, size)
        return artifacts

    def _publish(
        self,
        run_id: str,
        node_id: str,
        fingerprint: str,
        sources: dict[str, Artifact],
        work: Path,
        final: bool,
    ) -> tuple[dict[str, Artifact], bool]:
        """Return artifacts and whether this attempt created their directory."""
        relative = (
            f"outputs/{run_id}"
            if final
            else f"artifacts/runs/{run_id}/{node_id}/{uuid.uuid4().hex}"
        )
        destination = self.storage.path(relative)
        if destination.exists():
            # A killed process may have renamed a complete final result before
            # committing its checkpoint. Accept only an intact matching journal.
            if final:
                recovered = self._recover_publication(
                    destination, run_id, node_id, fingerprint, sources
                )
                if recovered is not None:
                    return recovered, False
            raise ExecutionError(
                "Publication destination already exists; refusing to overwrite"
            )
        staging = work / f"publication-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        staged = {}
        for name, source in sources.items():
            # Flat, output-name-based filenames avoid collisions and unsafe paths.
            if not valid_identifier(name):
                raise ExecutionError("Module output names must be ASCII identifiers")
            extension = source.path.suffix
            target = staging / f"{name}{extension}"
            target.touch(mode=0o600)
            shutil.copyfile(source.path, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            if digest_file(target) != (source.sha256, source.size):
                raise ExecutionError("An output changed while being published")
            staged[name] = Artifact(target, source.kind, source.sha256, source.size)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        published = {
            name: Artifact(
                destination / item.path.name, item.kind, item.sha256, item.size
            )
            for name, item in staged.items()
        }
        if final:
            marker = staging / ".publication.json"
            marker.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "node_id": node_id,
                        "fingerprint": fingerprint,
                        "outputs": json.loads(
                            self.storage.serialize_artifacts(published)
                        ),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            with marker.open("rb") as stream:
                os.fsync(stream.fileno())
        staging.rename(destination)
        return published, True

    def _recover_publication(
        self,
        destination: Path,
        run_id: str,
        node_id: str,
        fingerprint: str,
        sources: dict[str, Artifact],
    ) -> dict[str, Artifact] | None:
        try:
            marker = self.storage.path(
                destination.relative_to(self.storage.root) / ".publication.json"
            )
            journal = json.loads(marker.read_text(encoding="utf-8"))
            recovered = self.storage.verified_artifacts(
                json.dumps(journal["outputs"]),
                {name: source.kind for name, source in sources.items()},
            )
            if (
                journal["run_id"] == run_id
                and journal["node_id"] == node_id
                and journal["fingerprint"] == fingerprint
                and recovered is not None
                and all(
                    artifact.path.parent == destination
                    and (artifact.sha256, artifact.size)
                    == (sources[name].sha256, sources[name].size)
                    for name, artifact in recovered.items()
                )
            ):
                return recovered
        except (OSError, ValueError, TypeError, KeyError):
            pass
        return None
