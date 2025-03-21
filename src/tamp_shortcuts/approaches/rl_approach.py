"""A pure RL approach."""

from dataclasses import dataclass
from typing import Generic, cast

import numpy as np
import torch
from gymnasium.core import ActType, ObsType
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_shortcuts.approaches.base import BaseApproach
from tamp_shortcuts.benchmarks.base import SceneSpec, Simulator, SimulatorEnvironment


@dataclass
class RLConfig:
    """Configuration for RL policy."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    ent_coef: float = 0.01
    device: str = "cuda"
    total_timesteps: int = 10_000
    training_record_interval: int = 100


class RLApproach(
    BaseApproach[ObsType, ActType, SceneSpec], Generic[ObsType, ActType, SceneSpec]
):
    """A pure RL approach."""

    def __init__(
        self,
        simulator: Simulator[ObsType, ActType, SceneSpec],
        seed: int,
        config: RLConfig,
    ) -> None:
        super().__init__(simulator, seed)
        self._config = config

        # Create an environment from the simulator
        self._env = SimulatorEnvironment(simulator)
        self._env.reset(seed=seed)

        # Create the PPO model
        self._ppo_model = PPO(
            "MlpPolicy",
            self._env,
            learning_rate=self._config.learning_rate,
            batch_size=self._config.batch_size,
            n_epochs=self._config.n_epochs,
            gamma=self._config.gamma,
            ent_coef=self._config.ent_coef,
            device=self._config.device,
        )

    def train(self) -> None:
        callback = TrainingProgressCallback(
            check_freq=self._config.training_record_interval
        )
        self._ppo_model.learn(
            total_timesteps=self._config.total_timesteps, callback=callback
        )

    def reset(self, init_obs: ObsType) -> None:
        pass

    def step(self, obs: ObsType) -> ActType:
        obs_arr = np.array(obs, dtype=np.float32)
        with torch.no_grad():
            action, _ = self._ppo_model.predict(obs_arr, deterministic=True)
        return cast(ActType, action)


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress."""

    def __init__(self, check_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:
            # Episode finished - record metrics
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)

            # Reset counters
            self.current_length = 0
            self.current_reward = 0.0

            # Print progress regularly
            n_episodes = len(self.success_history)
            if n_episodes % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                recent_rewards = self.episode_rewards[-self.check_freq :]

                print("\nTraining Progress:")
                print(f"Episodes: {n_episodes}")
                print(
                    f"Recent Success%: {sum(recent_successes)/len(recent_successes):.2%}"
                )
                print(f"Recent Avg Episode Length: {np.mean(recent_lengths):.2f}")
                print(f"Recent Avg Reward: {np.mean(recent_rewards):.2f}")

        return True

    def _on_training_end(self) -> None:
        """Print final training statistics."""
        print("\nFinal Training Results:")
        if self.success_history:
            print(f"Overall Success Rate: {self._get_success_rate:.2%}")
            print(f"Overall Avg Episode Length: {self._get_avg_episode_length:.2f}")
            print(f"Overall Avg Reward: {self._get_avg_reward:.2f}")
        else:
            print("No episodes completed during training.")

    @property
    def _get_success_rate(self) -> float:
        """Get the success rate over all training."""
        if not self.success_history:
            return 0.0
        return float(sum(self.success_history) / len(self.success_history))

    @property
    def _get_avg_episode_length(self) -> float:
        """Get the average episode length over all training."""
        if not self.episode_lengths:
            return 0.0
        return float(np.mean(self.episode_lengths))

    @property
    def _get_avg_reward(self) -> float:
        """Get the average reward over all training."""
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))
