"""
Training script for the Conveyor System RL environment using Stable-Baselines3.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from environment import ConveyorEnv


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    save_path: str = "models",
    log_path: str = "logs",
):
    """Train a PPO agent on the conveyor environment."""

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Create vectorized training environments
    env_kwargs = {
        "spawn_rate_a": 0.8,
        "spawn_rate_b": 0.8,
        "max_steps": 200,
        "zipper_bonus": 2.0,
        "consecutive_penalty": 0.5,
    }
    train_env = make_vec_env(
        ConveyorEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
    )

    # Create evaluation environment
    eval_env = Monitor(ConveyorEnv(**env_kwargs))

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=log_path,
        eval_freq=10000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="conveyor_ppo",
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        },
        verbose=1,
        tensorboard_log=log_path,
    )

    print("Starting training...")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Save path: {save_path}")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_path, "conveyor_ppo_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")

    return model


def evaluate(model_path: str, n_episodes: int = 10):
    """Evaluate a trained model and print statistics."""

    model = PPO.load(model_path)
    env = ConveyorEnv(spawn_rate_a=0.8, spawn_rate_b=0.8, max_steps=200,
                      zipper_bonus=2.0, consecutive_penalty=0.5)

    total_rewards = []
    total_throughputs = []
    total_zipper_ratios = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_throughputs.append(info["throughput"])
        total_zipper_ratios.append(info["zipper_ratio"])

    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Mean reward: {sum(total_rewards) / n_episodes:.2f}")
    print(f"  Mean throughput: {sum(total_throughputs) / n_episodes:.1f}")
    print(f"  Mean zipper ratio: {sum(total_zipper_ratios) / n_episodes:.1%}")
    print(f"  Max throughput: {max(total_throughputs)}")
    print(f"  Min throughput: {min(total_throughputs)}")

    return total_rewards, total_throughputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train conveyor RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--eval", type=str, help="Path to model for evaluation (skip training)")

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
    else:
        model = train(total_timesteps=args.timesteps, n_envs=args.n_envs)
        evaluate("models/conveyor_ppo_final")
