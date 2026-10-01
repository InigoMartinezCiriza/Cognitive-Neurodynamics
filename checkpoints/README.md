# Trained agents

Agents at the end of the last curriculum stage of each run
(`<run>_<last stage>/`). Each directory contains a `tf.train.Checkpoint` with
the actor, the critic and their optimizers, and the sparsity masks of the
actor's hidden layer (`stage<k>_actor_layer0_{kernel,recur}.npy`).

Load one with

```python
from bioinspired_rnn.checkpoints import restore_weights
from bioinspired_rnn.config import load_config
from bioinspired_rnn.training import make_agent, make_env

config = load_config("EC_F_rnn")
env = make_env(config, stage=config.n_stages)
agent = make_agent(config, env)
restore_weights(agent, env.observation_space.shape[0], env.action_space.n,
                "checkpoints/EC_F_rnn_9", stage=9)
```

or run it with `python scripts/evaluate.py EC_F_rnn`.
