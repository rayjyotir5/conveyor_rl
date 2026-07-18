#!/usr/bin/env python3
"""
Main entry point for the Conveyor System RL project.

Usage:
    python run.py train          # Train the zipper-merge policy
    python run.py visualize      # Visualize a trained policy
    python run.py compare        # Compare trained vs random policy
    python run.py demo           # Quick demo with fewer training steps
    python run.py network-train  # Train on a conveyor network graph
    python run.py network-demo   # Short network throughput demo
"""

import argparse
import os
import sys


def _add_network_graph_args(parser):
    """Shared --preset / --graph options for network subcommands."""
    parser.add_argument(
        "--preset",
        type=str,
        default="merge",
        choices=["merge", "diamond", "triple"],
        help="Built-in network graph (default: merge)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Path to custom graph JSON (overrides --preset)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Conveyor System RL - Train and visualize merging policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train --timesteps 500000    # Full zipper-merge training
  python run.py demo                         # Quick demo (50k steps)
  python run.py visualize --model models/conveyor_ppo_final
  python run.py compare                      # Compare trained vs random
  python run.py network-demo --preset diamond
  python run.py network-train --graph graphs/custom_merge.json --timesteps 20000
  python run.py network-graph --preset merge --output network_graph.png
  python run.py network-visualize --model models/network/network_ppo_final --graph-output topology.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument("--timesteps", type=int, default=500_000,
                             help="Total training timesteps (default: 500000)")
    train_parser.add_argument("--n-envs", type=int, default=8,
                             help="Number of parallel environments (default: 8)")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize policy rollout")
    viz_parser.add_argument("--model", type=str, default="models/conveyor_ppo_final",
                           help="Path to trained model")
    viz_parser.add_argument("--output", type=str, default="rollout.gif",
                           help="Output file path (default: rollout.gif)")
    viz_parser.add_argument("--steps", type=int, default=200,
                           help="Simulation steps (default: 200)")
    viz_parser.add_argument("--fps", type=int, default=4,
                           help="Animation FPS (default: 4)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare trained vs random policy")
    compare_parser.add_argument("--model", type=str, default="models/conveyor_ppo_final",
                               help="Path to trained model")
    compare_parser.add_argument("--steps", type=int, default=200,
                               help="Simulation steps (default: 200)")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Quick demo with shorter training")
    demo_parser.add_argument("--timesteps", type=int, default=50_000,
                            help="Training timesteps for demo (default: 50000)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, default="models/conveyor_ppo_final",
                            help="Path to trained model")
    eval_parser.add_argument("--episodes", type=int, default=10,
                            help="Number of evaluation episodes (default: 10)")

    # Network graph commands
    net_train = subparsers.add_parser(
        "network-train", help="Train a small policy on a conveyor network graph"
    )
    _add_network_graph_args(net_train)
    net_train.add_argument("--timesteps", type=int, default=30_000,
                           help="Training timesteps (default: 30000)")
    net_train.add_argument("--n-envs", type=int, default=4,
                           help="Parallel environments (default: 4)")
    net_train.add_argument("--save-path", type=str, default="models/network",
                           help="Model save directory (default: models/network)")

    net_eval = subparsers.add_parser(
        "network-evaluate", help="Evaluate a network throughput policy"
    )
    _add_network_graph_args(net_eval)
    net_eval.add_argument("--model", type=str, default="models/network/network_ppo_final",
                          help="Path to trained network model")
    net_eval.add_argument("--episodes", type=int, default=10,
                          help="Evaluation episodes (default: 10)")
    net_eval.add_argument("--baseline", action="store_true",
                          help="Also report random-action baseline")

    net_graph = subparsers.add_parser(
        "network-graph", help="Draw a static diagram of the conveyor network graph"
    )
    _add_network_graph_args(net_graph)
    net_graph.add_argument("--output", type=str, default="network_graph.png",
                           help="Output PNG path (default: network_graph.png)")

    net_viz = subparsers.add_parser(
        "network-visualize", help="Visualize a network policy rollout (GIF)"
    )
    _add_network_graph_args(net_viz)
    net_viz.add_argument("--model", type=str, default="models/network/network_ppo_final",
                         help="Path to trained network model")
    net_viz.add_argument("--output", type=str, default="network_rollout.gif",
                         help="Output GIF path")
    net_viz.add_argument("--graph-output", type=str, default=None,
                         help="Also save a static graph PNG to this path")
    net_viz.add_argument("--steps", type=int, default=200,
                         help="Simulation steps (default: 200)")
    net_viz.add_argument("--fps", type=int, default=4,
                         help="Animation FPS (default: 4)")
    net_viz.add_argument("--random", action="store_true",
                         help="Use random actions instead of a trained model")

    net_demo = subparsers.add_parser(
        "network-demo", help="Train a small network policy and evaluate/visualize"
    )
    _add_network_graph_args(net_demo)
    net_demo.add_argument("--timesteps", type=int, default=15_000,
                          help="Training timesteps (default: 15000)")
    net_demo.add_argument("--output", type=str, default="network_demo.gif",
                          help="Output GIF path")
    net_demo.add_argument("--graph-output", type=str, default="network_graph.png",
                          help="Static graph PNG path (default: network_graph.png)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.command == "train":
        from train import train, evaluate
        print("=" * 60)
        print("CONVEYOR SYSTEM RL - TRAINING")
        print("=" * 60)
        train(total_timesteps=args.timesteps, n_envs=args.n_envs)
        print("\nRunning final evaluation...")
        evaluate("models/conveyor_ppo_final")

    elif args.command == "visualize":
        from visualize import visualize_trained_model
        print("=" * 60)
        print("CONVEYOR SYSTEM RL - VISUALIZATION")
        print("=" * 60)
        visualize_trained_model(args.model, args.output, args.steps, args.fps)
        print(f"\nVisualization saved to: {args.output}")

    elif args.command == "compare":
        from visualize import compare_policies
        print("=" * 60)
        print("CONVEYOR SYSTEM RL - POLICY COMPARISON")
        print("=" * 60)
        compare_policies(args.model, n_steps=args.steps)
        print("\nComparison visualizations saved to: comparisons/")

    elif args.command == "demo":
        from train import train, evaluate
        from visualize import visualize_trained_model, compare_policies
        print("=" * 60)
        print("CONVEYOR SYSTEM RL - QUICK DEMO")
        print("=" * 60)
        print(f"\nTraining for {args.timesteps} timesteps...")
        train(total_timesteps=args.timesteps, n_envs=4)
        print("\nEvaluating...")
        evaluate("models/conveyor_ppo_final")
        print("\nCreating visualization...")
        visualize_trained_model("models/conveyor_ppo_final", "demo_rollout.gif")
        print("\nComparing with random policy...")
        compare_policies("models/conveyor_ppo_final", output_dir="demo_comparison")
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("  - models/conveyor_ppo_final.zip (trained model)")
        print("  - demo_rollout.gif (policy visualization)")
        print("  - demo_comparison/ (trained vs random comparison)")

    elif args.command == "evaluate":
        from train import evaluate
        print("=" * 60)
        print("CONVEYOR SYSTEM RL - EVALUATION")
        print("=" * 60)
        evaluate(args.model, n_episodes=args.episodes)

    elif args.command == "network-train":
        from train_network import evaluate_network, train_network
        print("=" * 60)
        print("CONVEYOR NETWORK RL - TRAINING")
        print("=" * 60)
        train_network(
            preset=args.preset,
            graph_path=args.graph,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.save_path,
        )
        print("\nRunning evaluation...")
        evaluate_network(
            os.path.join(args.save_path, "network_ppo_final"),
            preset=args.preset,
            graph_path=args.graph,
        )

    elif args.command == "network-evaluate":
        from train_network import evaluate_network, evaluate_random_baseline
        print("=" * 60)
        print("CONVEYOR NETWORK RL - EVALUATION")
        print("=" * 60)
        evaluate_network(
            args.model,
            preset=args.preset,
            graph_path=args.graph,
            n_episodes=args.episodes,
        )
        if args.baseline:
            evaluate_random_baseline(
                preset=args.preset,
                graph_path=args.graph,
                n_episodes=args.episodes,
            )

    elif args.command == "network-graph":
        from visualize_network import render_graph
        print("=" * 60)
        print("CONVEYOR NETWORK RL - GRAPH DIAGRAM")
        print("=" * 60)
        path = render_graph(
            output_path=args.output,
            preset=args.preset,
            graph_path=args.graph,
        )
        print(f"\nGraph diagram saved to: {path}")

    elif args.command == "network-visualize":
        from visualize_network import visualize_network_model
        print("=" * 60)
        print("CONVEYOR NETWORK RL - ROLLOUT VISUALIZATION")
        print("=" * 60)
        visualize_network_model(
            None if args.random else args.model,
            output_path=args.output,
            preset=args.preset,
            graph_path=args.graph,
            n_steps=args.steps,
            fps=args.fps,
            graph_output=args.graph_output,
            random_policy=args.random,
        )
        print(f"\nRollout GIF saved to: {args.output}")
        if args.graph_output:
            print(f"Graph diagram saved to: {args.graph_output}")

    elif args.command == "network-demo":
        from train_network import evaluate_network, evaluate_random_baseline, train_network
        from visualize_network import visualize_network_model
        save_path = "models/network"
        print("=" * 60)
        print("CONVEYOR NETWORK RL - QUICK DEMO")
        print("=" * 60)
        print(f"\nTraining for {args.timesteps} timesteps on preset/graph...")
        train_network(
            preset=args.preset,
            graph_path=args.graph,
            total_timesteps=args.timesteps,
            n_envs=4,
            save_path=save_path,
        )
        model_path = os.path.join(save_path, "network_ppo_final")
        print("\nEvaluating trained policy...")
        evaluate_network(model_path, preset=args.preset, graph_path=args.graph)
        print("\nRandom baseline...")
        evaluate_random_baseline(preset=args.preset, graph_path=args.graph)
        print("\nCreating graph diagram + rollout visualization...")
        visualize_network_model(
            model_path,
            output_path=args.output,
            preset=args.preset,
            graph_path=args.graph,
            graph_output=args.graph_output,
        )
        print("\n" + "=" * 60)
        print("NETWORK DEMO COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print(f"  - {model_path}.zip")
        print(f"  - {args.graph_output}")
        print(f"  - {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
