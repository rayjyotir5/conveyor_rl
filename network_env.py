"""
Graph-based conveyor network RL environment.

Nodes: sources (spawn items), junctions (hold 1 item, route), sinks (exit).
Edges: conveyors of fixed length. Index 0 is nearest the target node.

Reward is pure throughput: +1.0 each time an item exits into a sink.
Optional small congestion penalty discourages jammed belts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_graphs import get_preset, load_graph, validate_graph


class NetworkConveyorEnv(gym.Env):
    """Throughput-optimizing conveyor network on an arbitrary small graph."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        graph: Optional[Dict[str, Any]] = None,
        preset: str = "merge",
        max_steps: int = 200,
        congestion_penalty: float = 0.01,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if graph is None:
            graph = get_preset(preset)
        else:
            validate_graph(graph)

        self.graph = graph
        self.max_steps = max_steps
        self.congestion_penalty = congestion_penalty
        self.render_mode = render_mode

        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edge_specs = list(graph["edges"])
        self.edge_ids = [e["id"] for e in self.edge_specs]
        self.edge_index = {eid: i for i, eid in enumerate(self.edge_ids)}

        self.source_ids = [n["id"] for n in graph["nodes"] if n["type"] == "source"]
        self.junction_ids = [n["id"] for n in graph["nodes"] if n["type"] == "junction"]
        self.sink_ids = [n["id"] for n in graph["nodes"] if n["type"] == "sink"]

        # Inbound / outbound edges per node (by edge id)
        self.inbound: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        self.outbound: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for e in self.edge_specs:
            self.outbound[e["source"]].append(e["id"])
            self.inbound[e["target"]].append(e["id"])

        # Observation: occupancy of every conveyor cell + junction buffers +
        # normalized selected input/output indices for each junction
        self._cell_slices: List[Tuple[str, int, int]] = []  # (edge_id, start, length)
        obs_parts = 0
        for e in self.edge_specs:
            length = e["length"]
            self._cell_slices.append((e["id"], obs_parts, length))
            obs_parts += length

        self._junction_obs_offset = obs_parts
        # per junction: has_item, in_sel_norm, out_sel_norm
        obs_parts += 3 * len(self.junction_ids)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_parts,), dtype=np.float32
        )

        # Actions: move flag per edge, then for each junction:
        #   inbound choice (max(1, n_in)), outbound choice (max(1, n_out))
        action_nvec: List[int] = [2] * len(self.edge_specs)
        self._junction_action_map: List[Tuple[str, int, int]] = []  # (jid, in_dim, out_dim)
        for jid in self.junction_ids:
            n_in = max(1, len(self.inbound[jid]))
            n_out = max(1, len(self.outbound[jid]))
            self._junction_action_map.append((jid, n_in, n_out))
            action_nvec.extend([n_in, n_out])

        self.action_space = spaces.MultiDiscrete(action_nvec)

        # Runtime state
        self.conveyors: Dict[str, List[Optional[Tuple[str, int]]]] = {}
        self.junction_item: Dict[str, Optional[Tuple[str, int]]] = {}
        self.junction_in_sel: Dict[str, int] = {}
        self.junction_out_sel: Dict[str, int] = {}

        self.current_step = 0
        self.total_throughput = 0
        self.item_counter = 0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ API
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.conveyors = {
            e["id"]: [None] * e["length"] for e in self.edge_specs
        }
        self.junction_item = {jid: None for jid in self.junction_ids}
        self.junction_in_sel = {jid: 0 for jid in self.junction_ids}
        self.junction_out_sel = {jid: 0 for jid in self.junction_ids}

        self.current_step = 0
        self.total_throughput = 0
        self.item_counter = 0
        self.history = []

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.int64)
        reward = 0.0
        exited_this_step = 0

        n_edges = len(self.edge_specs)
        move_flags = action[:n_edges]
        j_actions = action[n_edges:]

        # Decode junction routing selections
        cursor = 0
        for jid, n_in, n_out in self._junction_action_map:
            in_choice = int(j_actions[cursor]) % max(1, len(self.inbound[jid]) or 1)
            out_choice = int(j_actions[cursor + 1]) % max(1, len(self.outbound[jid]) or 1)
            self.junction_in_sel[jid] = in_choice
            self.junction_out_sel[jid] = out_choice
            cursor += 2

        # 1) Spawn at sources onto outbound conveyors (entry = last index)
        for sid in self.source_ids:
            rate = float(self.nodes[sid].get("spawn_rate", 0.5))
            for eid in self.outbound[sid]:
                belt = self.conveyors[eid]
                if belt[-1] is None and self.np_random.random() < rate:
                    self.item_counter += 1
                    belt[-1] = (sid, self.item_counter)

        # 2) Move conveyors whose move flag is set (toward target / index 0)
        for i, e in enumerate(self.edge_specs):
            if not move_flags[i]:
                continue
            eid = e["id"]
            belt = self.conveyors[eid]
            target = e["target"]

            # Item at front may exit into sink or wait for junction pull
            if belt[0] is not None and target in self.sink_ids:
                # Deliver to sink
                belt[0] = None
                self.total_throughput += 1
                exited_this_step += 1
                reward += 1.0

            # Shift toward front
            for pos in range(len(belt) - 1):
                if belt[pos] is None and belt[pos + 1] is not None:
                    belt[pos] = belt[pos + 1]
                    belt[pos + 1] = None

        # 3) Junction transfers: pull from selected inbound, push to selected outbound
        for jid in self.junction_ids:
            in_edges = self.inbound[jid]
            out_edges = self.outbound[jid]

            # Push first if possible (free buffer for pull)
            if self.junction_item[jid] is not None and out_edges:
                out_eid = out_edges[self.junction_out_sel[jid] % len(out_edges)]
                out_belt = self.conveyors[out_eid]
                if out_belt[-1] is None:
                    out_belt[-1] = self.junction_item[jid]
                    self.junction_item[jid] = None

            # Pull from selected inbound if buffer empty
            if self.junction_item[jid] is None and in_edges:
                in_eid = in_edges[self.junction_in_sel[jid] % len(in_edges)]
                in_belt = self.conveyors[in_eid]
                if in_belt[0] is not None:
                    self.junction_item[jid] = in_belt[0]
                    in_belt[0] = None

        # Light congestion penalty: occupied fraction of all cells
        if self.congestion_penalty > 0:
            total_cells = 0
            occupied = 0
            for belt in self.conveyors.values():
                total_cells += len(belt)
                occupied += sum(1 for c in belt if c is not None)
            occupied += sum(1 for v in self.junction_item.values() if v is not None)
            total_cells += max(len(self.junction_ids), 1)
            reward -= self.congestion_penalty * (occupied / total_cells)

        self.history.append(self._snapshot(action, reward, exited_this_step))

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return self._get_obs(), float(reward), False, truncated, self._get_info()

    # --------------------------------------------------------------- helpers
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for eid, start, length in self._cell_slices:
            belt = self.conveyors[eid]
            for i in range(length):
                obs[start + i] = 1.0 if belt[i] is not None else 0.0

        for j, jid in enumerate(self.junction_ids):
            base = self._junction_obs_offset + 3 * j
            obs[base] = 1.0 if self.junction_item[jid] is not None else 0.0
            n_in = max(1, len(self.inbound[jid]))
            n_out = max(1, len(self.outbound[jid]))
            obs[base + 1] = self.junction_in_sel[jid] / n_in
            obs[base + 2] = self.junction_out_sel[jid] / n_out
        return obs

    def _get_info(self) -> Dict[str, Any]:
        items_on_belts = sum(
            1 for belt in self.conveyors.values() for c in belt if c is not None
        )
        items_in_junctions = sum(
            1 for v in self.junction_item.values() if v is not None
        )
        return {
            "throughput": self.total_throughput,
            "step": self.current_step,
            "items_in_system": items_on_belts + items_in_junctions,
            "graph": self.graph.get("name", "custom"),
        }

    def _snapshot(self, action, reward, exited) -> Dict[str, Any]:
        return {
            "conveyors": {eid: belt.copy() for eid, belt in self.conveyors.items()},
            "junction_item": dict(self.junction_item),
            "junction_in_sel": dict(self.junction_in_sel),
            "junction_out_sel": dict(self.junction_out_sel),
            "action": np.array(action, copy=True),
            "throughput": self.total_throughput,
            "exited": exited,
            "reward": reward,
        }

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        return None

    def _render_text(self):
        print(f"=== Network step {self.current_step} throughput={self.total_throughput} ===")
        for eid, belt in self.conveyors.items():
            cells = "".join("[X]" if c else "[ ]" for c in belt)
            print(f"  {eid}: {cells}")
        for jid in self.junction_ids:
            item = "X" if self.junction_item[jid] else " "
            print(
                f"  junction {jid}: [{item}] "
                f"in={self.junction_in_sel[jid]} out={self.junction_out_sel[jid]}"
            )
        print("-" * 40)


def make_network_env(
    preset: Optional[str] = "merge",
    graph_path: Optional[str] = None,
    **kwargs,
) -> NetworkConveyorEnv:
    """Factory used by training / CLI."""
    if graph_path:
        graph = load_graph(graph_path)
        return NetworkConveyorEnv(graph=graph, **kwargs)
    return NetworkConveyorEnv(preset=preset, **kwargs)


gym.register(
    id="NetworkConveyor-v0",
    entry_point="network_env:NetworkConveyorEnv",
)
