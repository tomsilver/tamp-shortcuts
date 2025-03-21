"""Number line toy environment for testing."""

from dataclasses import dataclass
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete
from numpy.typing import NDArray

from tamp_shortcuts.benchmarks.base import Simulator


class NumberState(NamedTuple):
    """State of the number environment."""

    num: int  # location on the number line
    light_switch: bool  # whether the light is on


class NumberAction(NamedTuple):
    """Action in the number environment."""

    move: bool  # stay or move forward
    set_light: bool  # toggle light switch


@dataclass(frozen=True)
class SceneSpecNumber:
    """Scene specification for the number environment."""

    max_number: int = 2


class NumberEnvSimulator(Simulator[NumberState, NumberAction, SceneSpecNumber]):
    """Number environment in 1D."""

    @property
    def observation_space(self) -> gym.Space:
        return MultiDiscrete([self.scene_spec.max_number + 1, 2])

    @property
    def action_space(self) -> gym.Space:
        return MultiDiscrete([2, 2])

    def sample_initial_state(self, rng: np.random.Generator) -> NumberState:
        return NumberState(num=0, light_switch=False)

    def get_next_state(self, state: NumberState, action: NumberAction) -> NumberState:
        move_action, light_action = action

        new_light = bool(light_action)
        new_num = min(state.num + move_action, self.scene_spec.max_number)
        return NumberState(num=new_num, light_switch=new_light)

    def get_reward(self, state: NumberState, action: NumberAction) -> float:
        if self.check_done(state):
            return 1.0
        return -0.01

    def check_done(self, state: NumberState) -> bool:
        return state.light_switch and (state.num == self.scene_spec.max_number)

    def render_state(self, state: NumberState) -> NDArray[np.uint8]:
        raise NotImplementedError
