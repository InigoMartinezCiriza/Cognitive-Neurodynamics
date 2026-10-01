"""Behaviour of the environments."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from bioinspired_rnn.envs import EconomicChoiceEnv, make_cartpole  # noqa: E402
from bioinspired_rnn.envs.economic_choice import ACT_CHOOSE_LEFT, ACT_FIXATE  # noqa: E402


def run_until_decision(env):
    """Fixates until the decision epoch; returns the observations before it and
    the first observation of the decision epoch."""
    obs, info = env.reset(seed=0)
    observations = []
    while info["epoch"] != "Decision":
        observations.append(obs)
        obs, reward, terminated, truncated, info = env.step(ACT_FIXATE)
        assert not terminated and reward == pytest.approx(env.R_fix_step)
    return np.array(observations), obs


@pytest.mark.parametrize("partial", [False, True])
def test_economic_choice_observations(partial):
    env = EconomicChoiceEnv(duration_params=[100, 100, 100, 100], partial=partial, reward_B=1)
    observations, decision_obs = run_until_decision(env)

    assert np.all(observations[:, 0] == 1.0)            # fixation cue until the decision
    offer_obs = observations[-1]
    assert offer_obs[1] in (-1.0, 1.0)                  # juice position visible in the offer
    if partial:
        assert np.all(decision_obs == 0.0)              # offer hidden during the decision
    else:
        np.testing.assert_allclose(decision_obs[1:], offer_obs[1:])
        assert decision_obs[0] == 0.0


def test_choice_ends_trial_with_offer_reward():
    env = EconomicChoiceEnv(duration_params=[100, 100, 100, 100], reward_B=1)
    run_until_decision(env)
    _, reward, terminated, _, info = env.step(ACT_CHOOSE_LEFT)
    assert terminated
    assert reward == pytest.approx(env.trial_rL)
    assert info["chosen_action"] == ACT_CHOOSE_LEFT


def test_breaking_fixation_aborts():
    env = EconomicChoiceEnv()
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(ACT_CHOOSE_LEFT)
    assert terminated and reward == pytest.approx(env.R_ABORTED)


def test_partial_cartpole_observes_position_and_angle():
    env = make_cartpole(partial=True)
    obs, _ = env.reset(seed=0)
    full = env.unwrapped.state
    assert obs.shape == (2,)
    np.testing.assert_allclose(obs, np.asarray(full)[[0, 2]], rtol=1e-6)
