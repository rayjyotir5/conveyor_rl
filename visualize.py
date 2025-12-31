"""
Visualization of policy rollouts for the Conveyor System.

Creates animated visualizations showing items flowing through the conveyors
and turntable, with metrics overlay to see emergent behaviors like zipper merges.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
from stable_baselines3 import PPO
from environment import ConveyorEnv
from typing import List, Dict, Optional
import os


class ConveyorVisualizer:
    """Visualize conveyor system rollouts with matplotlib animations."""

    # Colors
    COLOR_EMPTY = "#E0E0E0"
    COLOR_ITEM_A = "#FF6B6B"  # Red for input A
    COLOR_ITEM_B = "#4ECDC4"  # Teal for input B
    COLOR_TURNTABLE = "#FFE66D"  # Yellow
    COLOR_CONVEYOR = "#95A5A6"
    COLOR_ACTIVE = "#27AE60"  # Green for active connection

    def __init__(self, env: ConveyorEnv):
        self.env = env
        self.length = env.conveyor_length

    def rollout(self, model: Optional[PPO] = None, n_steps: int = 200, deterministic: bool = True) -> List[Dict]:
        """Run a rollout and collect states for visualization."""
        obs, _ = self.env.reset()
        self.env.history = []  # Clear history

        for _ in range(n_steps):
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Random policy for comparison
                action = self.env.action_space.sample()

            obs, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                break

        return self.env.history

    def create_animation(
        self,
        history: List[Dict],
        output_path: str = "rollout.gif",
        fps: int = 4,
        figsize: tuple = (14, 8),
    ):
        """Create an animated visualization of the rollout."""

        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        ax_main = axes[0]
        ax_metrics = axes[1]

        # Setup main visualization
        ax_main.set_xlim(-1, 16)
        ax_main.set_ylim(-1, 6)
        ax_main.set_aspect('equal')
        ax_main.axis('off')
        ax_main.set_title("Conveyor System - Policy Rollout", fontsize=14, fontweight='bold')

        # Metrics tracking
        throughputs = []
        zipper_count = 0
        last_source = None

        def draw_conveyor(ax, x_start, y, items, direction='right', label=''):
            """Draw a conveyor belt with items."""
            patches_list = []

            for i in range(self.length):
                if direction == 'right':
                    x = x_start + i
                else:
                    x = x_start - i

                # Draw cell
                rect = patches.FancyBboxPatch(
                    (x, y), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=self.COLOR_EMPTY,
                    edgecolor=self.COLOR_CONVEYOR,
                    linewidth=2,
                )
                patches_list.append(ax.add_patch(rect))

                # Draw item if present
                item = items[i] if direction == 'right' else items[self.length - 1 - i]
                if item is not None:
                    color = self.COLOR_ITEM_A if item[0] == 'A' else self.COLOR_ITEM_B
                    item_circle = patches.Circle(
                        (x + 0.45, y + 0.45), 0.35,
                        facecolor=color,
                        edgecolor='white',
                        linewidth=2,
                    )
                    patches_list.append(ax.add_patch(item_circle))

            # Label
            if label:
                if direction == 'right':
                    ax.text(x_start - 0.5, y + 0.45, label, ha='right', va='center',
                           fontsize=10, fontweight='bold')
                else:
                    ax.text(x_start + 0.5, y + 0.45, label, ha='left', va='center',
                           fontsize=10, fontweight='bold')

            return patches_list

        def draw_turntable(ax, x, y, item, connected_to, transferring=False):
            """Draw the turntable."""
            patches_list = []

            # Turntable body
            turntable = patches.FancyBboxPatch(
                (x, y), 1.5, 1.5,
                boxstyle="round,pad=0.1",
                facecolor=self.COLOR_TURNTABLE,
                edgecolor='#D4A017',
                linewidth=3,
            )
            patches_list.append(ax.add_patch(turntable))

            # Connection indicator
            conn_y = y + 1.1 if connected_to == 0 else y + 0.4
            conn = patches.Circle(
                (x + 0.3, conn_y), 0.15,
                facecolor=self.COLOR_ACTIVE,
                edgecolor='white',
            )
            patches_list.append(ax.add_patch(conn))

            # Item on turntable
            if item is not None:
                color = self.COLOR_ITEM_A if item[0] == 'A' else self.COLOR_ITEM_B
                item_circle = patches.Circle(
                    (x + 0.75, y + 0.75), 0.4,
                    facecolor=color,
                    edgecolor='white',
                    linewidth=2,
                )
                patches_list.append(ax.add_patch(item_circle))

            # Label
            ax.text(x + 0.75, y - 0.3, 'Turntable', ha='center', va='top',
                   fontsize=9, style='italic')

            return patches_list

        def draw_arrows(ax, connected_to):
            """Draw flow arrows."""
            arrow_props = dict(arrowstyle='->', color='#7F8C8D', lw=2)
            active_arrow_props = dict(arrowstyle='->', color=self.COLOR_ACTIVE, lw=3)

            # Input A arrow
            props = active_arrow_props if connected_to == 0 else arrow_props
            ax.annotate('', xy=(6.2, 4.45), xytext=(5.1, 4.45), arrowprops=props)

            # Input B arrow
            props = active_arrow_props if connected_to == 1 else arrow_props
            ax.annotate('', xy=(6.2, 2.45), xytext=(5.1, 2.45), arrowprops=props)

            # Output arrow
            ax.annotate('', xy=(9, 3.45), xytext=(7.9, 3.45), arrowprops=arrow_props)

        def animate(frame):
            ax_main.clear()
            ax_metrics.clear()

            ax_main.set_xlim(-1, 17)
            ax_main.set_ylim(0, 6)
            ax_main.set_aspect('equal')
            ax_main.axis('off')

            state = history[frame]

            # Draw conveyors
            # Input A (top) - items flow left to right (index 4 is entry, 0 is exit to turntable)
            draw_conveyor(ax_main, 0, 4, list(reversed(state['conveyor_a'])), 'right', 'Input A')

            # Input B (bottom)
            draw_conveyor(ax_main, 0, 2, list(reversed(state['conveyor_b'])), 'right', 'Input B')

            # Output (right side)
            draw_conveyor(ax_main, 9, 3, state['conveyor_out'], 'right', '')
            ax_main.text(14.5, 3.45, 'Output', ha='left', va='center', fontsize=10, fontweight='bold')

            # Draw turntable
            draw_turntable(ax_main, 6, 2.75, state['turntable_item'], state['turntable_connected'])

            # Draw arrows
            draw_arrows(ax_main, state['turntable_connected'])

            # Action display
            action = state['action']
            action_text = f"Actions: A={'Move' if action[0] else 'Stop'}, B={'Move' if action[1] else 'Stop'}, " \
                         f"Out={'Move' if action[2] else 'Stop'}, Turntable={'→A' if action[3]==0 else '→B' if action[3]==1 else 'Transfer'}"
            ax_main.text(8, 5.5, action_text, ha='center', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Title with step info
            ax_main.set_title(f"Conveyor System - Step {frame+1}/{len(history)} | Throughput: {state['throughput']}",
                             fontsize=14, fontweight='bold')

            # Metrics plot
            throughputs.append(state['throughput'])
            ax_metrics.plot(throughputs, color='#3498DB', linewidth=2, label='Cumulative Throughput')
            ax_metrics.fill_between(range(len(throughputs)), throughputs, alpha=0.3, color='#3498DB')
            ax_metrics.set_xlabel('Step', fontsize=10)
            ax_metrics.set_ylabel('Items Processed', fontsize=10)
            ax_metrics.set_xlim(0, len(history))
            ax_metrics.set_ylim(0, max(max(throughputs) * 1.1, 10))
            ax_metrics.legend(loc='upper left')
            ax_metrics.grid(True, alpha=0.3)

            # Legend for item colors
            legend_elements = [
                patches.Patch(facecolor=self.COLOR_ITEM_A, edgecolor='white', label='From Input A'),
                patches.Patch(facecolor=self.COLOR_ITEM_B, edgecolor='white', label='From Input B'),
            ]
            ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

            plt.tight_layout()

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(history),
            interval=1000 // fps, blit=False
        )

        # Save
        print(f"Saving animation to {output_path}...")
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
        else:
            anim.save(output_path, writer='ffmpeg', fps=fps)

        plt.close()
        print(f"Animation saved!")

        return output_path

    def analyze_zipper_behavior(self, history: List[Dict]) -> Dict:
        """Analyze the rollout for zipper merge patterns."""
        if not history:
            return {'total_items': 0, 'items_from_a': 0, 'items_from_b': 0,
                    'alternations': 0, 'zipper_ratio': 0, 'final_throughput': 0}

        # Use the environment's tracking from the last frame
        last = history[-1]
        total_transfers = last.get('total_transfers', 0)
        alternations = last.get('total_alternations', 0)

        # Count items from each source
        items_a = sum(1 for s in history if s.get('transfer_source') == 'A')
        items_b = sum(1 for s in history if s.get('transfer_source') == 'B')

        # Zipper ratio: alternations / (transfers - 1)
        # Perfect zipper A-B-A-B has (n-1) alternations for n transfers
        zipper_ratio = alternations / max(total_transfers - 1, 1) if total_transfers > 1 else 0

        return {
            'total_items': total_transfers,
            'items_from_a': items_a,
            'items_from_b': items_b,
            'alternations': alternations,
            'zipper_ratio': zipper_ratio,
            'final_throughput': last['throughput'],
        }


def visualize_trained_model(
    model_path: str,
    output_path: str = "rollout.gif",
    n_steps: int = 200,
    fps: int = 4,
):
    """Load a trained model and create a visualization of its behavior."""

    model = PPO.load(model_path)
    env = ConveyorEnv(spawn_rate_a=0.8, spawn_rate_b=0.8, max_steps=n_steps,
                      zipper_bonus=2.0, consecutive_penalty=0.5)
    viz = ConveyorVisualizer(env)

    print("Running rollout...")
    history = viz.rollout(model, n_steps=n_steps)

    print("Analyzing behavior...")
    analysis = viz.analyze_zipper_behavior(history)
    print(f"\nBehavior Analysis:")
    print(f"  Total items processed: {analysis['total_items']}")
    print(f"  Items from A: {analysis['items_from_a']}")
    print(f"  Items from B: {analysis['items_from_b']}")
    print(f"  Source alternations: {analysis['alternations']}")
    print(f"  Zipper ratio: {analysis['zipper_ratio']:.2%}")
    print(f"  Final throughput: {analysis['final_throughput']}")

    viz.create_animation(history, output_path, fps=fps)

    return analysis


def compare_policies(model_path: str, output_dir: str = "comparisons", n_steps: int = 200):
    """Compare trained policy with random policy."""

    os.makedirs(output_dir, exist_ok=True)
    env_kwargs = {
        "spawn_rate_a": 0.8,
        "spawn_rate_b": 0.8,
        "max_steps": n_steps,
        "zipper_bonus": 2.0,
        "consecutive_penalty": 0.5,
    }

    # Trained policy
    model = PPO.load(model_path)
    env = ConveyorEnv(**env_kwargs)
    viz = ConveyorVisualizer(env)

    print("Visualizing trained policy...")
    history_trained = viz.rollout(model, n_steps=n_steps)
    viz.create_animation(history_trained, os.path.join(output_dir, "trained_policy.gif"))
    analysis_trained = viz.analyze_zipper_behavior(history_trained)

    # Random policy
    print("\nVisualizing random policy...")
    env2 = ConveyorEnv(**env_kwargs)
    viz2 = ConveyorVisualizer(env2)
    history_random = viz2.rollout(None, n_steps=n_steps)  # None = random policy
    viz2.create_animation(history_random, os.path.join(output_dir, "random_policy.gif"))
    analysis_random = viz2.analyze_zipper_behavior(history_random)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"{'Metric':<25} {'Trained':<15} {'Random':<15}")
    print("-"*55)
    print(f"{'Throughput':<25} {analysis_trained['final_throughput']:<15} {analysis_random['final_throughput']:<15}")
    print(f"{'Zipper Ratio':<25} {analysis_trained['zipper_ratio']*100:>6.1f}%         {analysis_random['zipper_ratio']*100:>6.1f}%")
    print(f"{'Items from A':<25} {analysis_trained['items_from_a']:<15} {analysis_random['items_from_a']:<15}")
    print(f"{'Items from B':<25} {analysis_trained['items_from_b']:<15} {analysis_random['items_from_b']:<15}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize conveyor policy rollouts")
    parser.add_argument("--model", type=str, default="models/conveyor_ppo_final",
                       help="Path to trained model")
    parser.add_argument("--output", type=str, default="rollout.gif",
                       help="Output file path")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of steps to simulate")
    parser.add_argument("--fps", type=int, default=4,
                       help="Animation frames per second")
    parser.add_argument("--compare", action="store_true",
                       help="Compare trained vs random policy")

    args = parser.parse_args()

    if args.compare:
        compare_policies(args.model, n_steps=args.steps)
    else:
        visualize_trained_model(args.model, args.output, args.steps, args.fps)
