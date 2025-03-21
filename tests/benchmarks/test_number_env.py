"""Tests for the NumberEnvSimulator class."""

import numpy as np
from gymnasium.spaces import MultiDiscrete

from tamp_shortcuts.benchmarks.number_env import (
    NumberAction,
    NumberEnvSimulator,
    NumberState,
    SceneSpecNumber,
)


def test_number_env_init():
    """Test initialization of NumberEnvSimulator."""
    scene_spec = SceneSpecNumber(max_number=5)
    simulator = NumberEnvSimulator(scene_spec)

    assert simulator.scene_spec.max_number == 5
    assert isinstance(simulator.observation_space, MultiDiscrete)
    assert np.array_equal(simulator.observation_space.nvec, [6, 2])
    assert isinstance(simulator.action_space, MultiDiscrete)
    assert np.array_equal(simulator.action_space.nvec, [2, 2])


def test_sample_initial_state():
    """Test sampling of initial state."""
    scene_spec = SceneSpecNumber()
    simulator = NumberEnvSimulator(scene_spec)
    rng = np.random.default_rng(0)

    state = simulator.sample_initial_state(rng)
    assert state.num == 0
    assert state.light_switch is False


def test_get_next_state():
    """Test state transitions."""
    scene_spec = SceneSpecNumber(max_number=3)
    simulator = NumberEnvSimulator(scene_spec)

    # Test moving forward
    state = NumberState(num=0, light_switch=False)
    action = NumberAction(move=True, set_light=False)
    next_state = simulator.get_next_state(state, action)
    assert next_state.num == 1
    assert next_state.light_switch is False

    # Test staying in place
    state = NumberState(num=1, light_switch=False)
    action = NumberAction(move=False, set_light=False)
    next_state = simulator.get_next_state(state, action)
    assert next_state.num == 1
    assert next_state.light_switch is False

    # Test turning on light
    state = NumberState(num=1, light_switch=False)
    action = NumberAction(move=False, set_light=True)
    next_state = simulator.get_next_state(state, action)
    assert next_state.num == 1
    assert next_state.light_switch is True

    # Test max number boundary
    state = NumberState(num=3, light_switch=False)
    action = NumberAction(move=True, set_light=False)
    next_state = simulator.get_next_state(state, action)
    assert next_state.num == 3  # Should not exceed max_number
    assert next_state.light_switch is False


def test_get_reward():
    """Test reward calculation."""
    scene_spec = SceneSpecNumber(max_number=2)
    simulator = NumberEnvSimulator(scene_spec)

    # Test non-terminal state
    state = NumberState(num=1, light_switch=False)
    action = NumberAction(move=True, set_light=False)
    reward = simulator.get_reward(state, action)
    assert reward == -0.01

    # Test terminal state
    state = NumberState(num=2, light_switch=True)
    action = NumberAction(move=False, set_light=True)
    reward = simulator.get_reward(state, action)
    assert reward == 1.0


def test_check_done():
    """Test done condition."""
    scene_spec = SceneSpecNumber(max_number=2)
    simulator = NumberEnvSimulator(scene_spec)

    # Test non-terminal states
    assert not simulator.check_done(NumberState(num=0, light_switch=False))
    assert not simulator.check_done(NumberState(num=2, light_switch=False))
    assert not simulator.check_done(NumberState(num=1, light_switch=True))

    # Test terminal state
    assert simulator.check_done(NumberState(num=2, light_switch=True))
