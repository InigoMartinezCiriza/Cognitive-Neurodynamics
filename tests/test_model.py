"""Modified GRU equations and a short end-to-end training run (needs TensorFlow)."""

import pickle

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bioinspired_rnn.config import load_config  # noqa: E402
from bioinspired_rnn.modified_gru import ModifiedGRU  # noqa: E402
from bioinspired_rnn.recording import critic_hidden_sequence, play_trial  # noqa: E402
from bioinspired_rnn.training import make_agent, make_env, run  # noqa: E402


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@pytest.mark.parametrize("run_name", ["EC_P_rnn", "EC_P_rnn_std"])
def test_recorded_critic_activity_is_recurrent(run_name):
    """The recorded critic states equal stepping the critic with its state carried over."""
    config = load_config(run_name)
    env = make_env(config, stage=1)
    agent = make_agent(config, env)
    actor_states, critic_states, _, _ = play_trial(agent, env)
    actions = np.random.default_rng(0).integers(0, 3, len(actor_states))

    sequence = critic_hidden_sequence(agent, actor_states, actions)
    hidden, stepped = None, []
    for state, action in zip(actor_states, actions):
        _, hidden = agent.evaluate_critic_step(state, np.eye(3)[action], critic_hidden_states=hidden)
        stepped.append(hidden[0].numpy().flatten())
    np.testing.assert_allclose(sequence, np.array(stepped), atol=1e-5)
    assert critic_states.shape == actor_states.shape


def test_modified_gru_matches_equations():
    alpha = 0.1
    layer = ModifiedGRU(4, alpha=alpha, return_sequences=True, return_state=True)
    x = np.random.default_rng(0).standard_normal((1, 5, 3)).astype(np.float32)
    outputs, _ = layer(x)

    c = layer.cell
    w = {name: getattr(c, name).numpy() for name in
         ("W_in", "W_rec", "b", "Wl_in", "Wl_rec", "bl", "Wg_in", "Wg_rec", "bg")}
    h = np.zeros(4)
    for t in range(5):
        lam = sigmoid(h @ w["Wl_rec"] + x[0, t] @ w["Wl_in"] + w["bl"])
        gam = sigmoid(h @ w["Wg_rec"] + x[0, t] @ w["Wg_in"] + w["bg"])
        candidate = (gam * h) @ w["W_rec"] + x[0, t] @ w["W_in"] + w["b"]
        h = np.maximum(0.0, h + alpha * lam * (candidate - h))
        np.testing.assert_allclose(outputs[0, t].numpy(), h, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("run_name", ["CP_F_rnn", "CP_P_rnn_std", "EC_P_ffnn"])
def test_two_stage_training(tmp_path, run_name):
    """Two stages of a few episodes: outputs are saved and stage 2 restores stage 1."""
    config = load_config(run_name)
    config.stages = config.stages[:2]
    run(config, tmp_path, max_episodes=2, print_interval=1000)

    for stage in (1, 2):
        with open(tmp_path / "Outputs" / f"{run_name}_{stage}.pkl", "rb") as f:
            rewards, actor_losses, critic_losses, actor_states, critic_states, records = pickle.load(f)
        assert len(rewards) == len(records) == 2
        recurrent = config.agent["layer_type"] != "Dense"
        assert (actor_states is not None) == recurrent
        assert (tmp_path / "Checkpoints" / f"{run_name}_{stage}" / "checkpoint").exists()
