"""Base approach."""

import abc
from typing import Generic

import numpy as np
from gymnasium.core import ActType, ObsType

from tamp_shortcuts.benchmarks.base import SceneSpec, Simulator


class BaseApproach(Generic[ObsType, ActType, SceneSpec]):
    """Base class for approaches."""

    def __init__(
        self, simulator: Simulator[ObsType, ActType, SceneSpec], seed: int
    ) -> None:
        self._simulator = simulator
        self._seed = 0
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def reset(self, init_obs: ObsType) -> None:
        """Reset the approach with the initial observation."""

    @abc.abstractmethod
    def step(self, obs: ObsType) -> ActType:
        """Take a step in the approach given the current observation."""
        raise NotImplementedError
