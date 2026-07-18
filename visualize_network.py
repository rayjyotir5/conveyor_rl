"""
Visualization for graph-based conveyor networks.

Provides:
  - Static topology diagrams (PNG) of sources / junctions / sinks / belts
  - Animated policy rollouts (GIF) with move/routing highlights + throughput
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from stable_baselines3 import PPO

from network_env import NetworkConveyorEnv, make_network_env


PRESET_LAYOUTS = {
    "merge_two_to_one": {
        "src_a": (0.0, 2.2),
        "src_b": (0.0, 0.0),
        "merge": (3.2, 1.1),
        "sink": (6.4, 1.1),
    },
    "diamond": {
        "src": (0.0, 1.1),
        "split": (2.2, 1.1),
        "merge": (5.4, 1.1),
        "sink": (7.6, 1.1),
    },
    "triple_merge": {
        "src_a": (0.0, 2.6),
        "src_b": (0.0, 1.1),
        "src_c": (0.0, -0.4),
        "merge": (3.2, 1.1),
        "sink": (6.4, 1.1),
    },
    "custom_merge": {
        "src_a": (0.0, 2.2),
        "src_b": (0.0, 0.0),
        "merge": (3.2, 1.1),
        "sink": (6.4, 1.1),
    },
    "custom_line": {
        "src": (0.0, 1.0),
        "mid": (3.0, 1.0),
        "sink": (6.0, 1.0),
    },
}


def _layout_positions(env: NetworkConveyorEnv) -> Dict[str, Tuple[float, float]]:
    name = env.graph.get("name", "")
    if name in PRESET_LAYOUTS:
        return dict(PRESET_LAYOUTS[name])

    # Layered fallback: sources left, sinks right, junctions in the middle
    sources = env.source_ids
    sinks = env.sink_ids
    junctions = env.junction_ids
    positions: Dict[str, Tuple[float, float]] = {}

    def place_column(ids: List[str], x: float):
        if not ids:
            return
        if len(ids) == 1:
            positions[ids[0]] = (x, 1.0)
            return
        for i, nid in enumerate(ids):
            y = 2.4 - i * (2.4 / (len(ids) - 1))
            positions[nid] = (x, y)

    place_column(sources, 0.0)
    place_column(junctions, 3.2)
    place_column(sinks, 6.4)

    # Any leftover nodes (shouldn't happen) go circular
    missing = [nid for nid in env.nodes if nid not in positions]
    for i, nid in enumerate(missing):
        angle = 2 * np.pi * i / max(len(missing), 1)
        positions[nid] = (2.0 * np.cos(angle), 2.0 * np.sin(angle))
    return positions


def _offset_endpoints(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    parallel_index: int,
    parallel_count: int,
    node_radius: float = 0.42,
) -> Tuple[float, float, float, float]:
    dx, dy = x1 - x0, y1 - y0
    norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / norm, dy / norm
    ox, oy = -uy, ux

    offset = 0.0
    if parallel_count > 1:
        offset = (parallel_index - (parallel_count - 1) / 2) * 0.55

    x0o = x0 + ux * node_radius + ox * offset
    y0o = y0 + uy * node_radius + oy * offset
    x1o = x1 - ux * node_radius + ox * offset
    y1o = y1 - uy * node_radius + oy * offset
    return x0o, y0o, x1o, y1o


def _edge_cell_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    length: int,
) -> List[Tuple[float, float]]:
    """Cell centers from source→target (display order)."""
    points = []
    for i in range(length):
        t = (i + 1) / (length + 1)
        points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return points


class NetworkVisualizer:
    COLOR_SOURCE = "#E74C3C"
    COLOR_JUNCTION = "#F1C40F"
    COLOR_SINK = "#27AE60"
    COLOR_EMPTY = "#ECF0F1"
    COLOR_ITEM = "#2980B9"
    COLOR_EDGE = "#95A5A6"
    COLOR_MOVE = "#27AE60"
    COLOR_ACTIVE_ROUTE = "#E67E22"
    COLOR_BG = "#FAFBFC"

    def __init__(self, env: NetworkConveyorEnv):
        self.env = env
        self.positions = _layout_positions(env)

        pair_counts: Dict[Tuple[str, str], int] = {}
        pair_seen: Dict[Tuple[str, str], int] = {}
        for e in env.edge_specs:
            key = (e["source"], e["target"])
            pair_counts[key] = pair_counts.get(key, 0) + 1

        self.edge_geom: Dict[str, Dict[str, Any]] = {}
        for e in env.edge_specs:
            key = (e["source"], e["target"])
            idx = pair_seen.get(key, 0)
            pair_seen[key] = idx + 1
            sx, sy = self.positions[e["source"]]
            tx, ty = self.positions[e["target"]]
            x0, y0, x1, y1 = _offset_endpoints(
                sx, sy, tx, ty, idx, pair_counts[key]
            )
            cells = _edge_cell_points(x0, y0, x1, y1, e["length"])
            self.edge_geom[e["id"]] = {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "cells": cells,
                "length": e["length"],
                "source": e["source"],
                "target": e["target"],
            }

    def _bounds(self, pad: float = 1.35) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        for geom in self.edge_geom.values():
            xs.extend([geom["x0"], geom["x1"]])
            ys.extend([geom["y0"], geom["y1"]])
        return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)

    def _node_color(self, ntype: str) -> str:
        if ntype == "source":
            return self.COLOR_SOURCE
        if ntype == "sink":
            return self.COLOR_SINK
        return self.COLOR_JUNCTION

    def _decode_moves(self, action: np.ndarray) -> Dict[str, bool]:
        moves = {}
        for i, eid in enumerate(self.env.edge_ids):
            moves[eid] = bool(int(action[i])) if i < len(action) else False
        return moves

    def _draw_topology(
        self,
        ax,
        state: Optional[Dict[str, Any]] = None,
        show_cells: bool = True,
        title: Optional[str] = None,
    ):
        xlim, ylim = self._bounds()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(self.COLOR_BG)

        moves = self._decode_moves(state["action"]) if state is not None else {}
        active_edges = set()
        if state is not None:
            for jid in self.env.junction_ids:
                in_edges = self.env.inbound[jid]
                out_edges = self.env.outbound[jid]
                if in_edges:
                    active_edges.add(in_edges[state["junction_in_sel"][jid] % len(in_edges)])
                if out_edges:
                    active_edges.add(out_edges[state["junction_out_sel"][jid] % len(out_edges)])

        # Edges
        for eid, geom in self.edge_geom.items():
            moving = moves.get(eid, False)
            routed = eid in active_edges
            if moving:
                color = self.COLOR_MOVE
                lw = 3.0
            elif routed and state is not None:
                color = self.COLOR_ACTIVE_ROUTE
                lw = 2.5
            else:
                color = self.COLOR_EDGE
                lw = 1.8

            ax.annotate(
                "",
                xy=(geom["x1"], geom["y1"]),
                xytext=(geom["x0"], geom["y0"]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    mutation_scale=14,
                ),
            )

            if show_cells:
                belt = None
                if state is not None:
                    belt = state["conveyors"][eid]
                    display_items = list(reversed(belt))
                else:
                    display_items = [None] * geom["length"]

                for (x, y), item in zip(geom["cells"], display_items):
                    occupied = item is not None
                    face = self.COLOR_ITEM if occupied else self.COLOR_EMPTY
                    edge_c = self.COLOR_MOVE if moving else ("#2C3E50" if occupied else self.COLOR_EDGE)
                    ax.add_patch(
                        patches.Circle(
                            (x, y),
                            0.16,
                            facecolor=face,
                            edgecolor=edge_c,
                            linewidth=1.6 if moving else 1.2,
                            zorder=3,
                        )
                    )

            # Edge label near midpoint
            mx = (geom["x0"] + geom["x1"]) / 2
            my = (geom["y0"] + geom["y1"]) / 2
            label = eid if state is None else f"{eid}{' ▶' if moving else ''}"
            ax.text(
                mx,
                my + 0.38,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#34495E",
                fontweight="bold" if moving else "normal",
                zorder=4,
            )

        # Nodes
        for nid, node in self.env.nodes.items():
            x, y = self.positions[nid]
            ntype = node["type"]
            color = self._node_color(ntype)
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x - 0.38, y - 0.38),
                    0.76,
                    0.76,
                    boxstyle="round,pad=0.06",
                    facecolor=color,
                    edgecolor="#2C3E50",
                    linewidth=2.0,
                    zorder=5,
                )
            )

            subtitle = ntype
            if ntype == "source":
                rate = node.get("spawn_rate", 0.5)
                subtitle = f"source  p={rate:.2f}"
            ax.text(x, y - 0.62, nid, ha="center", va="top", fontsize=10, fontweight="bold", zorder=6)
            ax.text(x, y - 0.88, subtitle, ha="center", va="top", fontsize=7, color="#566573", zorder=6)

            if ntype == "junction" and state is not None:
                item = state["junction_item"].get(nid)
                if item is not None:
                    ax.add_patch(
                        patches.Circle(
                            (x, y),
                            0.2,
                            facecolor=self.COLOR_ITEM,
                            edgecolor="white",
                            linewidth=1.8,
                            zorder=7,
                        )
                    )
                in_sel = state["junction_in_sel"].get(nid, 0)
                out_sel = state["junction_out_sel"].get(nid, 0)
                in_edges = self.env.inbound[nid]
                out_edges = self.env.outbound[nid]
                in_name = in_edges[in_sel % len(in_edges)] if in_edges else "-"
                out_name = out_edges[out_sel % len(out_edges)] if out_edges else "-"
                ax.text(
                    x,
                    y + 0.58,
                    f"←{in_name}  →{out_name}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=self.COLOR_ACTIVE_ROUTE,
                    fontweight="bold",
                    zorder=6,
                )
            elif ntype == "junction":
                ax.text(x, y, "◎", ha="center", va="center", fontsize=12, zorder=6)

        graph_name = self.env.graph.get("name", "custom")
        ax.set_title(title or f"Conveyor network: {graph_name}", fontsize=13, fontweight="bold", pad=10)

        legend = [
            patches.Patch(facecolor=self.COLOR_SOURCE, edgecolor="#2C3E50", label="Source"),
            patches.Patch(facecolor=self.COLOR_JUNCTION, edgecolor="#2C3E50", label="Junction"),
            patches.Patch(facecolor=self.COLOR_SINK, edgecolor="#2C3E50", label="Sink"),
            Line2D([0], [0], color=self.COLOR_MOVE, lw=3, label="Belt moving"),
            Line2D([0], [0], color=self.COLOR_ACTIVE_ROUTE, lw=2.5, label="Active route"),
            patches.Patch(facecolor=self.COLOR_ITEM, edgecolor="#2C3E50", label="Item"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.92)

    def save_graph_diagram(
        self,
        output_path: str = "network_graph.png",
        figsize: tuple = (11, 6),
        dpi: int = 140,
    ) -> str:
        """Save a static topology diagram (no simulation)."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLOR_BG)
        desc = self.env.graph.get("description", "")
        title = f"Graph topology: {self.env.graph.get('name', 'custom')}"
        if desc:
            title += f"\n{desc}"
        self._draw_topology(ax, state=None, show_cells=True, title=title)

        # Side annotation: edge lengths
        edge_lines = [
            f"{e['id']}: {e['source']} → {e['target']}  (len={e['length']})"
            for e in self.env.edge_specs
        ]
        ax.text(
            0.02,
            0.02,
            "Edges\n" + "\n".join(edge_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#BDC3C7"),
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=self.COLOR_BG)
        plt.close(fig)
        print(f"Graph diagram saved to {output_path}")
        return output_path

    def rollout(
        self,
        model: Optional[PPO] = None,
        n_steps: int = 200,
        deterministic: bool = True,
        seed: Optional[int] = 0,
    ) -> List[Dict]:
        obs, _ = self.env.reset(seed=seed)
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
        figsize: tuple = (12, 8),
    ) -> str:
        """Animate a policy/random rollout on the graph."""
        if not history:
            raise ValueError("Empty history — nothing to animate")

        fig = plt.figure(figsize=figsize, facecolor=self.COLOR_BG)
        gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.55, 1.0], hspace=0.28)
        ax_main = fig.add_subplot(gs[0])
        ax_actions = fig.add_subplot(gs[1])
        ax_metrics = fig.add_subplot(gs[2])

        throughputs: List[int] = []
        items_in_system: List[int] = []

        def _count_items(state: Dict[str, Any]) -> int:
            n = sum(1 for belt in state["conveyors"].values() for c in belt if c is not None)
            n += sum(1 for v in state["junction_item"].values() if v is not None)
            return n

        def animate(frame: int):
            ax_main.clear()
            ax_actions.clear()
            ax_metrics.clear()

            state = history[frame]
            title = (
                f"Policy rollout — {self.env.graph.get('name', 'custom')}  |  "
                f"Step {frame + 1}/{len(history)}  |  Throughput: {state['throughput']}"
            )
            self._draw_topology(ax_main, state=state, show_cells=True, title=title)

            # Action strip
            ax_actions.set_xlim(0, 1)
            ax_actions.set_ylim(0, 1)
            ax_actions.axis("off")
            moves = self._decode_moves(state["action"])
            move_txt = "  ".join(
                f"{eid}:{'MOVE' if moves[eid] else 'stop'}" for eid in self.env.edge_ids
            )
            route_bits = []
            for jid in self.env.junction_ids:
                in_edges = self.env.inbound[jid]
                out_edges = self.env.outbound[jid]
                in_sel = state["junction_in_sel"][jid]
                out_sel = state["junction_out_sel"][jid]
                in_name = in_edges[in_sel % len(in_edges)] if in_edges else "-"
                out_name = out_edges[out_sel % len(out_edges)] if out_edges else "-"
                route_bits.append(f"{jid}: {in_name}→{out_name}")
            route_txt = "  |  ".join(route_bits) if route_bits else "(no junctions)"
            ax_actions.text(
                0.5,
                0.62,
                f"Actions  {move_txt}",
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="#EAF2F8", edgecolor="#AED6F1"),
            )
            ax_actions.text(
                0.5,
                0.18,
                f"Routing  {route_txt}",
                ha="center",
                va="center",
                fontsize=9,
                family="monospace",
                color="#A04000",
                bbox=dict(boxstyle="round", facecolor="#FDEBD0", edgecolor="#F5B041"),
            )

            # Metrics
            throughputs.append(state["throughput"])
            items_in_system.append(_count_items(state))
            ax_metrics.plot(throughputs, color="#2980B9", linewidth=2.2, label="Cumulative throughput")
            ax_metrics.plot(items_in_system, color="#8E44AD", linewidth=1.6, alpha=0.85, label="Items in system")
            ax_metrics.fill_between(range(len(throughputs)), throughputs, alpha=0.15, color="#2980B9")
            ax_metrics.set_xlim(0, max(len(history) - 1, 1))
            ymax = max(max(throughputs + items_in_system) * 1.15, 5)
            ax_metrics.set_ylim(0, ymax)
            ax_metrics.set_xlabel("Step")
            ax_metrics.set_ylabel("Count")
            ax_metrics.grid(True, alpha=0.3)
            ax_metrics.legend(loc="upper left", fontsize=8)
            if state.get("exited"):
                ax_metrics.set_title(f"+{state['exited']} exited this step", fontsize=9, color=self.COLOR_SINK)

        anim = animation.FuncAnimation(
            fig, animate, frames=len(history), interval=max(1000 // fps, 50), blit=False
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        print(f"Saving network animation to {output_path}...")
        anim.save(output_path, writer="pillow", fps=fps)
        plt.close(fig)
        print("Animation saved!")
        return output_path


def render_graph(
    output_path: str = "network_graph.png",
    preset: str = "merge",
    graph_path: Optional[str] = None,
) -> str:
    """CLI helper: draw static graph topology."""
    env = make_network_env(
        preset=preset if graph_path is None else None,
        graph_path=graph_path,
        max_steps=1,
    )
    viz = NetworkVisualizer(env)
    return viz.save_graph_diagram(output_path)


def visualize_network_model(
    model_path: Optional[str] = None,
    output_path: str = "network_rollout.gif",
    preset: str = "merge",
    graph_path: Optional[str] = None,
    n_steps: int = 200,
    fps: int = 4,
    graph_output: Optional[str] = None,
    random_policy: bool = False,
    seed: int = 0,
) -> List[Dict]:
    """
    Run a rollout and write a GIF. Optionally also write a static graph PNG.

    If model_path is None or random_policy=True, uses random actions.
    """
    env = make_network_env(
        preset=preset if graph_path is None else None,
        graph_path=graph_path,
        max_steps=n_steps,
    )
    viz = NetworkVisualizer(env)

    if graph_output:
        viz.save_graph_diagram(graph_output)

    model = None
    if not random_policy:
        if model_path is None:
            raise ValueError("model_path required unless random_policy=True")
        model = PPO.load(model_path)

    label = "random" if model is None else "trained"
    print(f"Running {label} network rollout ({n_steps} steps)...")
    history = viz.rollout(model, n_steps=n_steps, seed=seed)
    final_tp = history[-1]["throughput"] if history else 0
    print(f"Final throughput: {final_tp}")
    viz.create_animation(history, output_path, fps=fps)
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize conveyor network graphs and rollouts")
    parser.add_argument("--preset", default="merge", choices=["merge", "diamond", "triple"])
    parser.add_argument("--graph", default=None, help="Custom graph JSON")
    parser.add_argument("--model", default=None, help="Trained model path")
    parser.add_argument("--output", default="network_rollout.gif")
    parser.add_argument("--graph-output", default=None, help="Also save static graph PNG")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--random", action="store_true", help="Random policy (no model)")
    parser.add_argument("--graph-only", action="store_true", help="Only write static graph PNG")
    args = parser.parse_args()

    if args.graph_only:
        out = args.graph_output or args.output.replace(".gif", ".png")
        if not out.endswith(".png"):
            out = "network_graph.png"
        render_graph(out, preset=args.preset, graph_path=args.graph)
    else:
        visualize_network_model(
            model_path=args.model,
            output_path=args.output,
            preset=args.preset,
            graph_path=args.graph,
            n_steps=args.steps,
            fps=args.fps,
            graph_output=args.graph_output,
            random_policy=args.random or args.model is None,
        )
