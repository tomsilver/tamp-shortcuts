"""Implementation of a random action selection approach."""

from typing import Generic

from gymnasium.core import ActType, ObsType

from tamp_shortcuts.approaches.base import BaseApproach
from tamp_shortcuts.benchmarks.base import SceneSpec, Simulator


class RandomActionsApproach(
    BaseApproach[ObsType, ActType, SceneSpec], Generic[ObsType, ActType, SceneSpec]
):
    """An approach that selects actions uniformly at random.

    This is the simplest possible approach to solving an environment - just pick actions
    randomly without any consideration for the current state.
    """

    def __init__(
        self, simulator: Simulator[ObsType, ActType, SceneSpec], seed: int
    ) -> None:
        super().__init__(simulator, seed)
        self._simulator.action_space.seed(seed)

    def reset(self, init_obs: ObsType) -> None:
        # No internal state to reset for random action selection
        pass

    def step(self, obs: ObsType) -> ActType:
        return self._simulator.action_space.sample()
