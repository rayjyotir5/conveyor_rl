"""
Visualization for graph-based conveyor networks.

Draws nodes/edges from layout heuristics and animates item flow + throughput.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from network_env import NetworkConveyorEnv, make_network_env


# Simple fixed layouts for presets; custom graphs get a circular layout.
PRESET_LAYOUTS = {
    "merge_two_to_one": {
        "src_a": (0.0, 2.0),
        "src_b": (0.0, 0.0),
        "merge": (3.0, 1.0),
        "sink": (6.0, 1.0),
    },
    "diamond": {
        "src": (0.0, 1.0),
        "split": (2.0, 1.0),
        "merge": (5.0, 1.0),
        "sink": (7.0, 1.0),
        # edge mid helpers unused; nodes only
    },
    "triple_merge": {
        "src_a": (0.0, 2.5),
        "src_b": (0.0, 1.0),
        "src_c": (0.0, -0.5),
        "merge": (3.0, 1.0),
        "sink": (6.0, 1.0),
    },
}


def _layout_positions(env: NetworkConveyorEnv) -> Dict[str, Tuple[float, float]]:
    name = env.graph.get("name", "")
    if name in PRESET_LAYOUTS:
        return dict(PRESET_LAYOUTS[name])

    # Circular layout for custom graphs
    ids = list(env.nodes.keys())
    n = len(ids)
    positions = {}
    for i, nid in enumerate(ids):
        angle = 2 * np.pi * i / max(n, 1) - np.pi / 2
        positions[nid] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))
    return positions


def _edge_points(
    positions: Dict[str, Tuple[float, float]],
    src: str,
    tgt: str,
    length: int,
    parallel_index: int = 0,
    parallel_count: int = 1,
) -> List[Tuple[float, float]]:
    """Sample cell centers along an edge, with offset for parallel edges."""
    x0, y0 = positions[src]
    x1, y1 = positions[tgt]
    dx, dy = x1 - x0, y1 - y0
    # Perpendicular offset for parallel belts (e.g. diamond top/bot)
    if parallel_count > 1:
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ox, oy = -dy / norm, dx / norm
        offset = (parallel_index - (parallel_count - 1) / 2) * 0.55
        x0, y0 = x0 + ox * offset, y0 + oy * offset
        x1, y1 = x1 + ox * offset, y1 + oy * offset

    # Keep endpoints clear of node bodies
    points = []
    for i in range(length):
        # index 0 near target, length-1 near source — match env indexing for drawing
        # Visual left-to-right: source→target, so display reversed index order
        t = (i + 1) / (length + 1)
        # display order from source to target: use display index
        points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return points


class NetworkVisualizer:
    COLOR_SOURCE = "#E74C3C"
    COLOR_JUNCTION = "#F1C40F"
    COLOR_SINK = "#2ECC71"
    COLOR_EMPTY = "#D5D8DC"
    COLOR_ITEM = "#3498DB"
    COLOR_EDGE = "#7F8C8D"

    def __init__(self, env: NetworkConveyorEnv):
        self.env = env
        self.positions = _layout_positions(env)

        # Parallel edge offsets keyed by (source, target)
        pair_counts: Dict[Tuple[str, str], int] = {}
        pair_seen: Dict[Tuple[str, str], int] = {}
        for e in env.edge_specs:
            key = (e["source"], e["target"])
            pair_counts[key] = pair_counts.get(key, 0) + 1
        self.edge_draw: Dict[str, List[Tuple[float, float]]] = {}
        for e in env.edge_specs:
            key = (e["source"], e["target"])
            idx = pair_seen.get(key, 0)
            pair_seen[key] = idx + 1
            # For drawing, list cells source→target (visual), env belt is target-near at 0
            pts = _edge_points(
                self.positions,
                e["source"],
                e["target"],
                e["length"],
                parallel_index=idx,
                parallel_count=pair_counts[key],
            )
            self.edge_draw[e["id"]] = pts

    def rollout(
        self,
        model: Optional[PPO] = None,
        n_steps: int = 200,
        deterministic: bool = True,
    ) -> List[Dict]:
        obs, _ = self.env.reset()
        self.env.history = []
        for _ in range(n_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        return self.env.history

    def create_animation(
        self,
        history: List[Dict],
        output_path: str = "network_rollout.gif",
        fps: int = 4,
        figsize: tuple = (12, 7),
    ) -> str:
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        ax_main, ax_metrics = axes
        throughputs: List[int] = []

        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        pad = 1.2
        xlim = (min(xs) - pad, max(xs) + pad)
        ylim = (min(ys) - pad, max(ys) + pad)

        def animate(frame: int):
            ax_main.clear()
            ax_metrics.clear()
            ax_main.set_xlim(*xlim)
            ax_main.set_ylim(*ylim)
            ax_main.set_aspect("equal")
            ax_main.axis("off")

            state = history[frame]

            # Edges / cells
            for e in self.env.edge_specs:
                eid = e["id"]
                pts = self.edge_draw[eid]
                belt = state["conveyors"][eid]
                # pts are source→target; belt[0] is nearest target → reverse for draw
                display_items = list(reversed(belt))
                for (x, y), item in zip(pts, display_items):
                    color = self.COLOR_ITEM if item is not None else self.COLOR_EMPTY
                    circ = patches.Circle(
                        (x, y),
                        0.18,
                        facecolor=color,
                        edgecolor=self.COLOR_EDGE,
                        linewidth=1.5,
                    )
                    ax_main.add_patch(circ)
                # edge label
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                ax_main.text(mx, my + 0.35, eid, ha="center", va="bottom", fontsize=8, color="#555")

            # Nodes
            for nid, node in self.env.nodes.items():
                x, y = self.positions[nid]
                ntype = node["type"]
                if ntype == "source":
                    color = self.COLOR_SOURCE
                elif ntype == "sink":
                    color = self.COLOR_SINK
                else:
                    color = self.COLOR_JUNCTION
                rect = patches.FancyBboxPatch(
                    (x - 0.35, y - 0.35),
                    0.7,
                    0.7,
                    boxstyle="round,pad=0.05",
                    facecolor=color,
                    edgecolor="#333",
                    linewidth=2,
                )
                ax_main.add_patch(rect)
                ax_main.text(x, y - 0.55, nid, ha="center", va="top", fontsize=9, fontweight="bold")

                if ntype == "junction":
                    item = state["junction_item"].get(nid)
                    if item is not None:
                        ax_main.add_patch(
                            patches.Circle(
                                (x, y),
                                0.2,
                                facecolor=self.COLOR_ITEM,
                                edgecolor="white",
                                linewidth=1.5,
                            )
                        )
                    in_sel = state["junction_in_sel"].get(nid, 0)
                    out_sel = state["junction_out_sel"].get(nid, 0)
                    ax_main.text(
                        x,
                        y + 0.5,
                        f"in:{in_sel} out:{out_sel}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#333",
                    )

            title = (
                f"Network ({self.env.graph.get('name', 'custom')}) — "
                f"Step {frame + 1}/{len(history)} | Throughput: {state['throughput']}"
            )
            ax_main.set_title(title, fontsize=13, fontweight="bold")

            legend = [
                patches.Patch(facecolor=self.COLOR_SOURCE, label="Source"),
                patches.Patch(facecolor=self.COLOR_JUNCTION, label="Junction"),
                patches.Patch(facecolor=self.COLOR_SINK, label="Sink"),
                patches.Patch(facecolor=self.COLOR_ITEM, label="Item"),
            ]
            ax_main.legend(handles=legend, loc="upper right", fontsize=8)

            throughputs.append(state["throughput"])
            ax_metrics.plot(throughputs, color="#2980B9", linewidth=2)
            ax_metrics.fill_between(range(len(throughputs)), throughputs, alpha=0.25, color="#2980B9")
            ax_metrics.set_xlim(0, len(history))
            ax_metrics.set_ylim(0, max(max(throughputs) * 1.1, 5))
            ax_metrics.set_xlabel("Step")
            ax_metrics.set_ylabel("Throughput")
            ax_metrics.grid(True, alpha=0.3)
            plt.tight_layout()

        anim = animation.FuncAnimation(
            fig, animate, frames=len(history), interval=1000 // fps, blit=False
        )
        print(f"Saving network animation to {output_path}...")
        anim.save(output_path, writer="pillow", fps=fps)
        plt.close()
        print("Animation saved!")
        return output_path


def visualize_network_model(
    model_path: str,
    output_path: str = "network_rollout.gif",
    preset: str = "merge",
    graph_path: Optional[str] = None,
    n_steps: int = 200,
    fps: int = 4,
):
    model = PPO.load(model_path)
    env = make_network_env(
        preset=preset if graph_path is None else None,
        graph_path=graph_path,
        max_steps=n_steps,
    )
    viz = NetworkVisualizer(env)
    print("Running network rollout...")
    history = viz.rollout(model, n_steps=n_steps)
    final_tp = history[-1]["throughput"] if history else 0
    print(f"Final throughput: {final_tp}")
    viz.create_animation(history, output_path, fps=fps)
    return history
