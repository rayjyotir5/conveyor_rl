# Conveyor System RL - Zipper Merge Learning

A reinforcement learning project that trains an AI agent to efficiently manage a 3-way conveyor system with two inputs merging through a turntable to one output. The agent learns to maximize throughput while discovering emergent behaviors like the **zipper merge** pattern.

## Demo

### Trained Policy - Zipper Merge Behavior
![Zipper Merge Demo](zipper_merge.gif)

The trained agent learns to alternate between input sources (A and B), creating an efficient zipper merge pattern that maximizes throughput.

### Policy Comparison

| Trained Policy | Random Policy |
|:--------------:|:-------------:|
| ![Trained](comparisons/trained_policy.gif) | ![Random](comparisons/random_policy.gif) |

The trained policy achieves significantly higher throughput and zipper ratio compared to random actions.

## Project Overview

### Environment

```
Input A [4][3][2][1][0] →
                          [Turntable] → [0][1][2][3][4] Output
Input B [4][3][2][1][0] →
```

- **Two input conveyors** (A and B) feed items toward a central turntable
- **Turntable** can connect to either input and transfer items to output
- **Output conveyor** carries items to the exit point
- Items spawn randomly on input conveyors based on configurable spawn rates

### Reward Structure

| Event | Reward |
|-------|--------|
| Item exits output conveyor | +1.0 |
| Source alternation (zipper) | +2.0 |
| Same source consecutively | -0.5 * consecutive_count |
| Blocking (item waiting, wrong turntable direction) | -0.05 |

### Action Space

The agent controls 4 discrete actions per step:

| Action | Values |
|--------|--------|
| Conveyor A | 0=Stop, 1=Move |
| Conveyor B | 0=Stop, 1=Move |
| Output Conveyor | 0=Stop, 1=Move |
| Turntable | 0=Connect A, 1=Connect B, 2=Transfer to Output |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/conveyor_rl.git
cd conveyor_rl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `gymnasium>=0.29.0` - RL environment framework
- `stable-baselines3>=2.0.0` - PPO algorithm implementation
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Visualization
- `pillow>=9.0.0` - GIF generation

## Usage

### Quick Start - Demo Mode

Run a quick demo that trains for 50k steps and generates visualizations:

```bash
python run.py demo
```

This will:
1. Train a model for 50,000 timesteps
2. Evaluate the trained model
3. Generate `demo_rollout.gif`
4. Create comparison visualizations in `demo_comparison/`

### Full Training

Train a model with the default 500k timesteps:

```bash
python run.py train
```

Customize training parameters:

```bash
python run.py train --timesteps 1000000 --n-envs 16
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 500,000 | Total training timesteps |
| `--n-envs` | 8 | Number of parallel environments |

### Visualization

Generate a visualization of a trained policy:

```bash
python run.py visualize --model models/conveyor_ppo_final
```

Options:

```bash
python run.py visualize --model models/conveyor_ppo_final --output my_rollout.gif --steps 300 --fps 6
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | models/conveyor_ppo_final | Path to trained model |
| `--output` | rollout.gif | Output file path |
| `--steps` | 200 | Simulation steps |
| `--fps` | 4 | Animation frames per second |

### Compare Policies

Compare a trained policy against random actions:

```bash
python run.py compare --model models/conveyor_ppo_final
```

This generates two GIFs in the `comparisons/` directory:
- `trained_policy.gif` - Trained agent behavior
- `random_policy.gif` - Random action baseline

### Evaluate Model

Run evaluation episodes and print statistics:

```bash
python run.py evaluate --model models/conveyor_ppo_final --episodes 20
```

## Project Structure

```
conveyor_rl/
├── run.py              # Main entry point with CLI commands
├── environment.py      # Gymnasium environment definition
├── train.py            # PPO training script
├── visualize.py        # Visualization and animation tools
├── requirements.txt    # Python dependencies
├── models/             # Saved model checkpoints
│   ├── best/           # Best model during training
│   ├── checkpoints/    # Periodic checkpoints
│   └── conveyor_ppo_final.zip
├── logs/               # TensorBoard logs
├── comparisons/        # Policy comparison GIFs
└── *.gif               # Generated visualizations
```

## Understanding the Visualizations

### Main Display

![Visualization Guide](demo_rollout.gif)

- **Red circles**: Items from Input A
- **Teal circles**: Items from Input B
- **Yellow box**: Turntable (green dot shows active connection)
- **Gray boxes**: Empty conveyor positions
- **Bottom graph**: Cumulative throughput over time

### Action Display

The top of each frame shows the current actions:
- `A=Move/Stop` - Input A conveyor movement
- `B=Move/Stop` - Input B conveyor movement
- `Out=Move/Stop` - Output conveyor movement
- `Turntable=→A/→B/Transfer` - Turntable connection or transfer action

## Training Details

### Algorithm

The project uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3:

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 3e-4 |
| Batch Size | 64 |
| N Steps | 2048 |
| N Epochs | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |
| Network Architecture | MLP [64, 64] |

### Monitoring Training

Training logs are saved for TensorBoard:

```bash
tensorboard --logdir logs
```

## Key Metrics

After training, the model is evaluated on:

- **Throughput**: Total items successfully processed
- **Zipper Ratio**: Percentage of source alternations (100% = perfect zipper)
- **Mean Reward**: Average episode reward

Example evaluation output:
```
Evaluation over 10 episodes:
  Mean reward: 245.32
  Mean throughput: 67.8
  Mean zipper ratio: 82.5%
  Max throughput: 73
  Min throughput: 61
```

## Customization

### Environment Parameters

Modify `environment.py` or pass kwargs when creating the environment:

```python
env = ConveyorEnv(
    conveyor_length=5,      # Length of each conveyor
    spawn_rate_a=0.8,       # Item spawn probability for A
    spawn_rate_b=0.8,       # Item spawn probability for B
    max_steps=200,          # Episode length
    zipper_bonus=2.0,       # Reward for alternating sources
    consecutive_penalty=0.5 # Penalty multiplier for same source
)
```

### Training Parameters

Modify `train.py` to adjust PPO hyperparameters or network architecture.

## License

MIT License

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the PPO implementation
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the RL environment framework
