"""Tests for the RLApproach with the NumberEnvSimulator."""

from tamp_shortcuts.approaches.rl_approach import RLApproach, RLConfig
from tamp_shortcuts.benchmarks.base import SimulatorEnvironment
from tamp_shortcuts.benchmarks.number_env import (
    NumberEnvSimulator,
    SceneSpecNumber,
)


def test_rl_approach():
    """Test the RLApproach in the NumberEnv."""
    seed = 123

    # Create a simulator.
    scene_spec = SceneSpecNumber(max_number=2)
    sim = NumberEnvSimulator(scene_spec)

    # Create a config suitable for testing, but the environment is so simple
    # that this should work consistently.
    config = RLConfig(
        learning_rate=1e-4,
        batch_size=32,
        n_epochs=1,
        gamma=0.99,
        ent_coef=0.01,
        device="cpu",  # Use CPU for testing
        total_timesteps=1000,  # Fewer steps for tests
        training_record_interval=10,
    )

    # Create the approach.
    approach = RLApproach(sim, seed, config)

    # Train the approach.
    approach.train()

    # Test the approach in a new environment.
    env = SimulatorEnvironment(NumberEnvSimulator(scene_spec))
    obs, _ = env.reset(seed=seed)
    approach.reset(obs)
    for _ in range(5):
        action = approach.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert reward <= 0.0
        assert not truncated
        if terminated:
            break
    else:
        assert False, "Failed to reach goal"
