"""
Train a small PPO policy to maximize throughput on a conveyor network graph.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from network_env import NetworkConveyorEnv, make_network_env


def _env_factory(preset: str, graph_path: Optional[str], max_steps: int, congestion_penalty: float):
    def _thunk():
        return make_network_env(
            preset=preset if graph_path is None else None,
            graph_path=graph_path,
            max_steps=max_steps,
            congestion_penalty=congestion_penalty,
        )

    return _thunk


def train_network(
    preset: str = "merge",
    graph_path: Optional[str] = None,
    total_timesteps: int = 30_000,
    n_envs: int = 4,
    max_steps: int = 200,
    congestion_penalty: float = 0.01,
    save_path: str = "models/network",
    log_path: str = "logs/network",
    net_arch=None,
) -> PPO:
    """Train a compact MLP policy for network throughput."""

    if net_arch is None:
        net_arch = dict(pi=[32, 32], vf=[32, 32])

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    env_kwargs: Dict[str, Any] = {
        "max_steps": max_steps,
        "congestion_penalty": congestion_penalty,
    }
    if graph_path:
        env_kwargs["graph"] = None  # loaded inside factory via path — use thunk instead
        train_env = make_vec_env(
            _env_factory(preset, graph_path, max_steps, congestion_penalty),
            n_envs=n_envs,
        )
        eval_env = Monitor(
            make_network_env(
                preset=None,
                graph_path=graph_path,
                max_steps=max_steps,
                congestion_penalty=congestion_penalty,
            )
        )
    else:
        env_kwargs["preset"] = preset
        train_env = make_vec_env(
            NetworkConveyorEnv,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
        )
        eval_env = Monitor(NetworkConveyorEnv(**env_kwargs))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best"),
        log_path=log_path,
        eval_freq=max(5_000 // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="network_ppo",
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": net_arch},
        verbose=1,
        tensorboard_log=log_path,
    )

    graph_label = graph_path or preset
    print("Starting network training...")
    print(f"  Graph: {graph_label}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Policy net: {net_arch}")
    print(f"  Save path: {save_path}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = os.path.join(save_path, "network_ppo_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}")
    return model


def evaluate_network(
    model_path: str,
    preset: str = "merge",
    graph_path: Optional[str] = None,
    n_episodes: int = 10,
    max_steps: int = 200,
    congestion_penalty: float = 0.01,
):
    """Evaluate a trained network policy and print throughput stats."""
    model = PPO.load(model_path)
    env = make_network_env(
        preset=preset if graph_path is None else None,
        graph_path=graph_path,
        max_steps=max_steps,
        congestion_penalty=congestion_penalty,
    )

    rewards = []
    throughputs = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
        throughputs.append(info["throughput"])

    print(f"\nNetwork evaluation over {n_episodes} episodes:")
    print(f"  Graph: {info.get('graph', preset)}")
    print(f"  Mean reward: {sum(rewards) / n_episodes:.2f}")
    print(f"  Mean throughput: {sum(throughputs) / n_episodes:.1f}")
    print(f"  Max throughput: {max(throughputs)}")
    print(f"  Min throughput: {min(throughputs)}")
    return rewards, throughputs


def evaluate_random_baseline(
    preset: str = "merge",
    graph_path: Optional[str] = None,
    n_episodes: int = 10,
    max_steps: int = 200,
):
    """Random-action baseline for comparison."""
    env = make_network_env(
        preset=preset if graph_path is None else None,
        graph_path=graph_path,
        max_steps=max_steps,
    )
    throughputs = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        throughputs.append(info["throughput"])
    print(f"\nRandom baseline over {n_episodes} episodes:")
    print(f"  Mean throughput: {sum(throughputs) / n_episodes:.1f}")
    print(f"  Max throughput: {max(throughputs)}")
    print(f"  Min throughput: {min(throughputs)}")
    return throughputs
