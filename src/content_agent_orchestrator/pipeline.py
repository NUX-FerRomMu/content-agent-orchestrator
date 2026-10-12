"""Parse and validate a DAG before creating execution state."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Module, PipelineError


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def valid_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value) is not None
    )


@dataclass(frozen=True)
class InputReference:
    external: str | None = None
    node: str | None = None
    output: str | None = None


@dataclass(frozen=True)
class Node:
    id: str
    module: str
    inputs: Mapping[str, InputReference]
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class Pipeline:
    nodes: tuple[Node, ...]
    final: str
    definition: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path, registry: Mapping[str, Module]) -> "Pipeline":
        try:
            definition = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"Cannot read pipeline JSON: {exc}") from exc
        return cls.parse(definition, registry)

    @classmethod
    def parse(cls, definition: Any, registry: Mapping[str, Module]) -> "Pipeline":
        if not isinstance(definition, dict) or set(definition) != {"nodes", "final"}:
            raise PipelineError("A pipeline requires exactly 'nodes' and 'final'")
        raw_nodes = definition["nodes"]
        final = definition["final"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise PipelineError("A pipeline needs at least one node")
        if not valid_identifier(final):
            raise PipelineError("Invalid final node identifier")
        nodes: dict[str, Node] = {}
        for raw in raw_nodes:
            if (
                not isinstance(raw, dict)
                or not {"id", "module", "inputs"} <= raw.keys()
                or raw.keys() - {"id", "module", "inputs", "parameters"}
            ):
                raise PipelineError("Invalid node fields")
            node_id, module_name = raw["id"], raw["module"]
            if not valid_identifier(node_id) or node_id in nodes:
                raise PipelineError("Node identifiers must be safe and unique")
            if not isinstance(module_name, str) or module_name not in registry:
                raise PipelineError(f"Unknown module: {module_name!r}")
            module = registry[module_name]
            if (
                module.name != module_name
                or not isinstance(module.version, str)
                or not module.version
            ):
                raise PipelineError(
                    "Module names and versions must identify their implementation"
                )
            if not module.output_kinds or any(
                not valid_identifier(name) or not isinstance(kind, str) or not kind
                for name, kind in (
                    *module.input_kinds.items(),
                    *module.output_kinds.items(),
                )
            ):
                raise PipelineError("Modules require named, typed artifact contracts")
            raw_inputs = raw["inputs"]
            if not isinstance(raw_inputs, dict) or set(raw_inputs) != set(
                module.input_kinds
            ):
                raise PipelineError(f"Incorrect inputs for node {node_id}")
            inputs = {}
            for name, reference in raw_inputs.items():
                if not isinstance(reference, dict):
                    raise PipelineError(f"Invalid input reference in {node_id}")
                if set(reference) == {"external"} and valid_identifier(
                    reference["external"]
                ):
                    inputs[name] = InputReference(external=reference["external"])
                elif set(reference) == {"node", "output"} and all(
                    valid_identifier(reference[key]) for key in ("node", "output")
                ):
                    inputs[name] = InputReference(**reference)
                else:
                    raise PipelineError(f"Invalid input reference in {node_id}")
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, dict):
                raise PipelineError(f"Parameters must be an object in {node_id}")
            try:
                canonical_json(parameters)
                module.validate_parameters(parameters)
            except (TypeError, ValueError) as exc:
                raise PipelineError(f"Invalid parameters in {node_id}: {exc}") from exc
            nodes[node_id] = Node(node_id, module_name, inputs, parameters)

        if final not in nodes:
            raise PipelineError("Final node does not exist")
        dependencies: dict[str, set[str]] = {}
        children: dict[str, set[str]] = {key: set() for key in nodes}
        external_kinds: dict[str, str] = {}
        for node in nodes.values():
            dependencies[node.id] = set()
            for name, reference in node.inputs.items():
                expected = registry[node.module].input_kinds[name]
                if reference.external:
                    previous = external_kinds.setdefault(reference.external, expected)
                    if previous != expected:
                        raise PipelineError(
                            "External input has conflicting artifact kinds"
                        )
                    continue
                if reference.node not in nodes:
                    raise PipelineError(f"Unknown dependency in {node.id}")
                source = nodes[reference.node]
                actual = registry[source.module].output_kinds.get(reference.output)
                if actual != expected:
                    raise PipelineError(f"Incompatible or missing output in {node.id}")
                dependencies[node.id].add(source.id)
                children[source.id].add(node.id)
        sinks = {key for key, value in children.items() if not value}
        if sinks != {final}:
            raise PipelineError(
                "Every node must lead to the single declared final node"
            )

        ordered = []
        remaining = {key: set(value) for key, value in dependencies.items()}
        while remaining:
            ready = [key for key, value in remaining.items() if not value]
            if not ready:
                raise PipelineError("Pipeline contains a cycle")
            for key in ready:
                ordered.append(nodes[key])
                remaining.pop(key)
            for value in remaining.values():
                value.difference_update(ready)
        # A cycle disconnected from the final node is rejected above or here.
        reachable = {final}
        pending = [final]
        while pending:
            for parent in dependencies[pending.pop()]:
                if parent not in reachable:
                    reachable.add(parent)
                    pending.append(parent)
        if reachable != set(nodes):
            raise PipelineError("Every node must be an ancestor of the final node")
        return cls(tuple(ordered), final, json.loads(canonical_json(definition)))
