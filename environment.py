"""
Conveyor System RL Environment

A 3-way conveyor system with two inputs merging through a turntable to one output.
Goal: Maximize throughput while learning efficient merging strategies (e.g., zipper merge).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ConveyorEnv(gym.Env):
    """
    Conveyor system environment with turntable merge.

    Layout:
        Input A [4][3][2][1][0] →
                                  [Turntable] → [0][1][2][3][4] Output
        Input B [4][3][2][1][0] →

    State: conveyor occupancies + turntable state + last source + consecutive count
    Actions: move/stop each conveyor, turntable control
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        conveyor_length: int = 5,
        spawn_rate_a: float = 0.5,
        spawn_rate_b: float = 0.5,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        zipper_bonus: float = 2.0,
        consecutive_penalty: float = 0.5,
    ):
        super().__init__()

        self.conveyor_length = conveyor_length
        self.spawn_rate_a = spawn_rate_a
        self.spawn_rate_b = spawn_rate_b
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.zipper_bonus = zipper_bonus
        self.consecutive_penalty = consecutive_penalty

        # State dimensions:
        # - 3 conveyors (length each)
        # - turntable_connected_to (1)
        # - turntable_has_item (1)
        # - last_source (1: 0=none, 0.5=A, 1=B)
        # - consecutive_same_source normalized (1)
        obs_dim = 3 * conveyor_length + 4
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: [conv_a_move, conv_b_move, conv_out_move, turntable_action]
        # turntable_action: 0=connect_A, 1=connect_B, 2=transfer_to_output
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 3])

        # Internal state
        self.conveyor_a = None
        self.conveyor_b = None
        self.conveyor_out = None
        self.turntable_item = None
        self.turntable_connected = None
        self.last_source = None
        self.consecutive_same = 0

        self.current_step = 0
        self.total_throughput = 0
        self.item_counter = 0
        self.alternations = 0
        self.total_transfers = 0

        self.history = []

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(3 * self.conveyor_length + 4, dtype=np.float32)

        for i, item in enumerate(self.conveyor_a):
            obs[i] = 1.0 if item is not None else 0.0

        offset = self.conveyor_length
        for i, item in enumerate(self.conveyor_b):
            obs[offset + i] = 1.0 if item is not None else 0.0

        offset = 2 * self.conveyor_length
        for i, item in enumerate(self.conveyor_out):
            obs[offset + i] = 1.0 if item is not None else 0.0

        obs[-4] = float(self.turntable_connected)
        obs[-3] = 1.0 if self.turntable_item is not None else 0.0

        # Encode last source: 0=none, 0.5=A, 1.0=B
        if self.last_source is None:
            obs[-2] = 0.0
        elif self.last_source == 'A':
            obs[-2] = 0.5
        else:
            obs[-2] = 1.0

        # Normalize consecutive count (cap at 5)
        obs[-1] = min(self.consecutive_same / 5.0, 1.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        zipper_ratio = self.alternations / max(self.total_transfers - 1, 1) if self.total_transfers > 1 else 0
        return {
            "throughput": self.total_throughput,
            "step": self.current_step,
            "items_on_a": sum(1 for x in self.conveyor_a if x is not None),
            "items_on_b": sum(1 for x in self.conveyor_b if x is not None),
            "items_on_out": sum(1 for x in self.conveyor_out if x is not None),
            "alternations": self.alternations,
            "total_transfers": self.total_transfers,
            "zipper_ratio": zipper_ratio,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.conveyor_a = [None] * self.conveyor_length
        self.conveyor_b = [None] * self.conveyor_length
        self.conveyor_out = [None] * self.conveyor_length
        self.turntable_item = None
        self.turntable_connected = 0
        self.last_source = None
        self.consecutive_same = 0

        self.current_step = 0
        self.total_throughput = 0
        self.item_counter = 0
        self.alternations = 0
        self.total_transfers = 0
        self._last_transfer_source = None
        self._last_transfer_alternated = False
        self.history = []

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        move_a, move_b, move_out, turntable_action = action
        reward = 0.0

        # Record state before actions
        state_before = {
            'conveyor_a': self.conveyor_a.copy(),
            'conveyor_b': self.conveyor_b.copy(),
            'conveyor_out': self.conveyor_out.copy(),
            'turntable_item': self.turntable_item,
            'turntable_connected': self.turntable_connected,
        }

        # 1. Spawn new items (higher rate to create pressure)
        if self.np_random.random() < self.spawn_rate_a:
            if self.conveyor_a[-1] is None:
                self.item_counter += 1
                self.conveyor_a[-1] = ('A', self.item_counter)

        if self.np_random.random() < self.spawn_rate_b:
            if self.conveyor_b[-1] is None:
                self.item_counter += 1
                self.conveyor_b[-1] = ('B', self.item_counter)

        # 2. Process turntable connection switch
        if turntable_action == 0:
            self.turntable_connected = 0
        elif turntable_action == 1:
            self.turntable_connected = 1

        # 3. Move output conveyor first (create space for new items)
        if move_out:
            if self.conveyor_out[-1] is not None:
                self.total_throughput += 1
                reward += 1.0  # Throughput reward
            # Shift items toward exit
            for i in range(self.conveyor_length - 1, 0, -1):
                self.conveyor_out[i] = self.conveyor_out[i - 1]
            self.conveyor_out[0] = None

        # 4. Transfer from turntable to output (if action is transfer)
        if turntable_action == 2 and self.turntable_item is not None and self.conveyor_out[0] is None:
            self.conveyor_out[0] = self.turntable_item
            self.turntable_item = None

        # 5. Transfer from input conveyor to turntable (KEY FOR ZIPPER!)
        if self.turntable_item is None:
            transferred = False
            source = None

            if self.turntable_connected == 0 and self.conveyor_a[0] is not None:
                self.turntable_item = self.conveyor_a[0]
                self.conveyor_a[0] = None
                source = 'A'
                transferred = True
            elif self.turntable_connected == 1 and self.conveyor_b[0] is not None:
                self.turntable_item = self.conveyor_b[0]
                self.conveyor_b[0] = None
                source = 'B'
                transferred = True

            if transferred:
                self.total_transfers += 1
                self._last_transfer_source = source  # For history tracking
                self._last_transfer_alternated = False

                # ZIPPER REWARD LOGIC
                if self.last_source is not None:
                    if source != self.last_source:
                        # Alternated! Big bonus
                        reward += self.zipper_bonus
                        self.alternations += 1
                        self.consecutive_same = 0
                        self._last_transfer_alternated = True
                    else:
                        # Same source again - penalty scales with consecutive count
                        self.consecutive_same += 1
                        penalty = self.consecutive_penalty * self.consecutive_same
                        reward -= penalty

                self.last_source = source
            else:
                self._last_transfer_source = None
                self._last_transfer_alternated = False

        # 6. Move input conveyors
        if move_a:
            for i in range(self.conveyor_length - 1):
                if self.conveyor_a[i] is None and self.conveyor_a[i + 1] is not None:
                    self.conveyor_a[i] = self.conveyor_a[i + 1]
                    self.conveyor_a[i + 1] = None

        if move_b:
            for i in range(self.conveyor_length - 1):
                if self.conveyor_b[i] is None and self.conveyor_b[i + 1] is not None:
                    self.conveyor_b[i] = self.conveyor_b[i + 1]
                    self.conveyor_b[i + 1] = None

        # Small penalty for blocking (items waiting at front)
        if self.conveyor_a[0] is not None and self.turntable_connected != 0:
            reward -= 0.05  # A has item but turntable pointed at B
        if self.conveyor_b[0] is not None and self.turntable_connected != 1:
            reward -= 0.05  # B has item but turntable pointed at A

        # Record for visualization
        self.history.append({
            'conveyor_a': self.conveyor_a.copy(),
            'conveyor_b': self.conveyor_b.copy(),
            'conveyor_out': self.conveyor_out.copy(),
            'turntable_item': self.turntable_item,
            'turntable_connected': self.turntable_connected,
            'action': action.copy() if hasattr(action, 'copy') else np.array(action),
            'throughput': self.total_throughput,
            'reward': reward,
            'last_source': self.last_source,
            'consecutive_same': self.consecutive_same,
            'transfer_source': self._last_transfer_source,  # Source of transfer this step (or None)
            'transfer_alternated': self._last_transfer_alternated,  # Was this an alternation?
            'total_alternations': self.alternations,
            'total_transfers': self.total_transfers,
        })

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        return None

    def _render_text(self):
        def cell(item):
            if item is None:
                return "[ ]"
            return f"[{item[0]}]"

        a_str = "".join(cell(x) for x in reversed(self.conveyor_a))
        b_str = "".join(cell(x) for x in reversed(self.conveyor_b))
        out_str = "".join(cell(x) for x in self.conveyor_out)

        t_item = "X" if self.turntable_item else " "
        t_conn = "A" if self.turntable_connected == 0 else "B"

        print(f"Input A:  {a_str} →")
        print(f"                      [{t_item}|{t_conn}]")
        print(f"Input B:  {b_str} →     → {out_str} →")
        print(f"Throughput: {self.total_throughput}  Step: {self.current_step}")
        print(f"Last: {self.last_source}  Consecutive: {self.consecutive_same}")
        print("-" * 50)


gym.register(
    id="ConveyorSystem-v0",
    entry_point="environment:ConveyorEnv",
)
