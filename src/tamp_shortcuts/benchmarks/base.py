"""Base class for environments."""

import abc
from typing import Any, Generic, TypeVar

import numpy as np
from gymnasium.core import ActType, Env, ObsType
from gymnasium.spaces import Space
from numpy.typing import NDArray

SceneSpec = TypeVar("SceneSpec")


class Simulator(Generic[ObsType, ActType, SceneSpec]):
    """A deterministic environment simulator."""

    def __init__(self, scene_spec: SceneSpec) -> None:
        self.scene_spec = scene_spec

    @property
    @abc.abstractmethod
    def observation_space(self) -> Space[ObsType]:
        """Return the observation space of the environment."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Space[ActType]:
        """Return the action space of the environment."""

    @abc.abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> ObsType:
        """Sample an initial state for the environment."""

    @abc.abstractmethod
    def get_next_state(self, state: ObsType, action: ActType) -> ObsType:
        """Get the next state given the current state and action."""

    @abc.abstractmethod
    def get_reward(self, state: ObsType, action: ActType) -> float:
        """Get the reward for taking an action in a given state."""

    @abc.abstractmethod
    def check_done(self, state: ObsType) -> bool:
        """Check if the environment is done."""

    @abc.abstractmethod
    def render_state(self, state: ObsType) -> NDArray[np.uint8]:
        """Render the environment given a state."""


class Environment(Env[ObsType, ActType], Generic[ObsType, ActType, SceneSpec]):
    """A deterministic environment."""

    def __init__(self, simulator: Simulator[ObsType, ActType, SceneSpec]) -> None:
        self._simulator = simulator
        self._current_state: ObsType | None = None

    @property
    def _observation_space(self) -> Space[ObsType]:
        """Return the observation space of the environment."""
        return self._simulator.observation_space

    @property
    def _action_space(self) -> Space[ActType]:
        """Return the action space of the environment."""
        return self._simulator.action_space

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        assert self._current_state is not None
        next_state = self._simulator.get_next_state(self._current_state, action)
        reward = self._simulator.get_reward(self._current_state, action)
        done = self._simulator.check_done(next_state)
        return next_state, reward, done, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        assert self._np_random is not None, "Call seed() before reset()"
        self._current_state = self._simulator.sample_initial_state(self._np_random)
        return self._current_state, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the environment."""
        assert self._current_state is not None
        return self._simulator.render_state(self._current_state)
