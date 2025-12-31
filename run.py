#!/usr/bin/env python3
"""
Main entry point for the Conveyor System RL project.

Usage:
    python run.py train          # Train the policy
    python run.py visualize      # Visualize a trained policy
    python run.py compare        # Compare trained vs random policy
    python run.py demo           # Quick demo with fewer training steps
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Conveyor System RL - Train and visualize merging policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train --timesteps 500000    # Full training run
  python run.py demo                         # Quick demo (50k steps)
  python run.py visualize --model models/conveyor_ppo_final
  python run.py compare                      # Compare trained vs random
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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
