"""
Preset and custom conveyor network graph definitions.

A graph is a dict with:
  - nodes: list of {id, type, ...}
      type: "source" | "junction" | "sink"
      sources may set spawn_rate (float in [0, 1])
  - edges: list of {id, source, target, length?}
      length defaults to 5; items flow source → target (index 0 nearest target)

Junctions hold at most one item and choose which inbound edge to pull from
and which outbound edge to push onto.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List


def merge_two_to_one(
    length: int = 4,
    spawn_rate_a: float = 0.7,
    spawn_rate_b: float = 0.7,
) -> Dict[str, Any]:
    """Classic 2-input merge → single sink (throughput-focused)."""
    return {
        "name": "merge_two_to_one",
        "description": "Two sources merge at a junction into one sink",
        "nodes": [
            {"id": "src_a", "type": "source", "spawn_rate": spawn_rate_a},
            {"id": "src_b", "type": "source", "spawn_rate": spawn_rate_b},
            {"id": "merge", "type": "junction"},
            {"id": "sink", "type": "sink"},
        ],
        "edges": [
            {"id": "conv_a", "source": "src_a", "target": "merge", "length": length},
            {"id": "conv_b", "source": "src_b", "target": "merge", "length": length},
            {"id": "conv_out", "source": "merge", "target": "sink", "length": length},
        ],
    }


def diamond(
    length: int = 3,
    spawn_rate: float = 0.85,
) -> Dict[str, Any]:
    """One source splits to two parallel paths, then merges to one sink."""
    return {
        "name": "diamond",
        "description": "Source → split → two paths → merge → sink",
        "nodes": [
            {"id": "src", "type": "source", "spawn_rate": spawn_rate},
            {"id": "split", "type": "junction"},
            {"id": "merge", "type": "junction"},
            {"id": "sink", "type": "sink"},
        ],
        "edges": [
            {"id": "in", "source": "src", "target": "split", "length": length},
            {"id": "top", "source": "split", "target": "merge", "length": length},
            {"id": "bot", "source": "split", "target": "merge", "length": length},
            {"id": "out", "source": "merge", "target": "sink", "length": length},
        ],
    }


def triple_merge(
    length: int = 3,
    spawn_rate: float = 0.55,
) -> Dict[str, Any]:
    """Three sources merge through a junction into one sink."""
    return {
        "name": "triple_merge",
        "description": "Three sources merge into one sink",
        "nodes": [
            {"id": "src_a", "type": "source", "spawn_rate": spawn_rate},
            {"id": "src_b", "type": "source", "spawn_rate": spawn_rate},
            {"id": "src_c", "type": "source", "spawn_rate": spawn_rate},
            {"id": "merge", "type": "junction"},
            {"id": "sink", "type": "sink"},
        ],
        "edges": [
            {"id": "conv_a", "source": "src_a", "target": "merge", "length": length},
            {"id": "conv_b", "source": "src_b", "target": "merge", "length": length},
            {"id": "conv_c", "source": "src_c", "target": "merge", "length": length},
            {"id": "conv_out", "source": "merge", "target": "sink", "length": length},
        ],
    }


PRESETS = {
    "merge": merge_two_to_one,
    "diamond": diamond,
    "triple": triple_merge,
}


def get_preset(name: str, **kwargs) -> Dict[str, Any]:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return deepcopy(PRESETS[name](**kwargs))


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    validate_graph(graph)
    return graph


def save_graph(graph: Dict[str, Any], path: str) -> None:
    validate_graph(graph)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
        f.write("\n")


def validate_graph(graph: Dict[str, Any]) -> None:
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError("Graph must have 'nodes' and 'edges'")

    nodes: List[Dict[str, Any]] = graph["nodes"]
    edges: List[Dict[str, Any]] = graph["edges"]

    if not nodes:
        raise ValueError("Graph needs at least one node")
    if not edges:
        raise ValueError("Graph needs at least one edge")

    ids = [n["id"] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate node ids")
    id_set = set(ids)

    edge_ids = [e["id"] for e in edges]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate edge ids")

    types = {}
    for n in nodes:
        t = n.get("type")
        if t not in ("source", "junction", "sink"):
            raise ValueError(f"Node {n.get('id')}: invalid type '{t}'")
        types[n["id"]] = t
        if t == "source" and "spawn_rate" not in n:
            n["spawn_rate"] = 0.5

    for e in edges:
        for key in ("id", "source", "target"):
            if key not in e:
                raise ValueError(f"Edge missing '{key}': {e}")
        if e["source"] not in id_set or e["target"] not in id_set:
            raise ValueError(f"Edge {e['id']} references unknown node")
        if e["source"] == e["target"]:
            raise ValueError(f"Edge {e['id']} is a self-loop")
        length = e.get("length", 5)
        if not isinstance(length, int) or length < 1:
            raise ValueError(f"Edge {e['id']}: length must be positive int")
        e.setdefault("length", length)

        # Sources only emit; sinks only receive
        if types[e["target"]] == "source":
            raise ValueError(f"Edge {e['id']} cannot target a source")
        if types[e["source"]] == "sink":
            raise ValueError(f"Edge {e['id']} cannot leave a sink")

    if not any(types[n] == "source" for n in types):
        raise ValueError("Graph needs at least one source")
    if not any(types[n] == "sink" for n in types):
        raise ValueError("Graph needs at least one sink")
