"""Partially observable CartPole: only cart position and pole angle are observed."""

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

# Indices of the CartPole observation kept by the wrapper:
# 0 = cart position, 1 = cart velocity, 2 = pole angle, 3 = pole angular velocity.
OBSERVED = [0, 2]


class CartPolePartialEnv(ObservationWrapper):
    """Removes the two velocities from the CartPole observation, so they have to
    be inferred from the history of positions and angles."""

    def __init__(self, env):
        super().__init__(env)
        low, high = self.env.observation_space.low, self.env.observation_space.high
        self.observation_space = Box(low=low[OBSERVED].astype(np.float32),
                                     high=high[OBSERVED].astype(np.float32),
                                     dtype=np.float32)

    def observation(self, obs):
        return obs[OBSERVED]


def make_cartpole(partial=False, env_name="CartPole-v1"):
    """Creates the full (``partial=False``) or partial CartPole environment."""
    env = gym.make(env_name)
    return CartPolePartialEnv(env) if partial else env
