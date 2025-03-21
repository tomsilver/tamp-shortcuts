"""Tests for the RandomActionsApproach approach with the NumberEnvSimulator."""

from unittest.mock import patch

import numpy as np

from tamp_shortcuts.approaches.random_actions import RandomActionsApproach
from tamp_shortcuts.benchmarks.base import SimulatorEnvironment
from tamp_shortcuts.benchmarks.number_env import (
    NumberAction,
    NumberEnvSimulator,
    NumberState,
    SceneSpecNumber,
)


def test_random_actions_approach():
    """Test RandomActionsApproach with the NumberEnvSimulator()."""
    scene_spec = SceneSpecNumber(max_number=2)
    simulator = NumberEnvSimulator(scene_spec)
    seed = 123
    approach = RandomActionsApproach(simulator, seed)
    approach.reset(NumberState(num=0, light_switch=False))

    # Mock the sample method to verify it's called
    with patch.object(
        simulator.action_space,
        "sample",
        return_value=NumberAction(move=True, set_light=False),
    ) as mock_sample:
        action = approach.step(NumberState(num=0, light_switch=False))

        # Check that sample was called
        mock_sample.assert_called_once()
        # Check that the returned action is the mocked value
        assert action == NumberAction(move=True, set_light=False)

    # Create an environment and take a number of actions
    env = SimulatorEnvironment(NumberEnvSimulator(scene_spec))
    obs = env.reset(seed=123)
    approach.reset(obs)
    for _ in range(10):
        action = approach.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert reward <= 0.0
        assert not truncated
        if terminated:
            break


def test_random_actions_reproducibility():
    """Test that the actions are reproducible with the same seed."""
    scene_spec = SceneSpecNumber(max_number=2)
    simulator1 = NumberEnvSimulator(scene_spec)
    simulator2 = NumberEnvSimulator(scene_spec)
    approach1 = RandomActionsApproach(simulator1, seed=123)
    approach2 = RandomActionsApproach(simulator2, seed=123)

    # Run multiple steps with both approaches
    num_steps = 10
    actions1 = []
    actions2 = []
    state = NumberState(num=0, light_switch=False)

    for _ in range(num_steps):
        actions1.append(approach1.step(state))
        actions2.append(approach2.step(state))

    # The actions should be identical
    for a1, a2 in zip(actions1, actions2):
        assert np.allclose(a1, a2)
