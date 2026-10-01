# Run configurations

One TOML file per run of the paper. A run trains one agent on one environment
through a sequence of curriculum stages; each stage continues from the
checkpoint of the previous one.

```toml
name = "EC_F_rnn"           # run name, prefix of the output files
task = "economic_choice"    # or "cartpole"
partial = false             # partial-observables variant of the environment
seed = 1

[agent]                     # bioinspired_rnn.actor_critic.ActorCriticAgent
layer_type = "GRU_modified" # "Dense" (FFNN), "GRU_standard" or "GRU_modified"
actor_hidden_size = 50
critic_hidden_size = 50
actor_prob_connection = 0.1
critic_prob_connection = 1.0
alpha = 0.1
noise_std = 0.0
learning_rate = 0.004

[training]                  # bioinspired_rnn.reinforce.train_agent
gamma = 1.0
l2_actor = 1e-4
l2_critic = 1e-4

[env]                       # environment parameters shared by all stages
dt = 10
# ...

[[stages]]                  # one block per stage, in order
episodes = 20000
duration_params = [10, 10, 20, 20]   # stage-specific environment parameters
input_noise_sigma = 0.0
```

`agent.learning_rate` is used for the whole run. A stage may also set its own
`learning_rate`, which is assigned to both optimizers after the checkpoint of
the previous stage is restored; none of the runs of the paper does it.

Run a configuration with `python scripts/train.py <name>`.

## Notes on the runs

The configurations describe the runs whose outputs are analysed in the paper;
the number of episodes and hidden units of every stage were checked against the
saved outputs.

* **Fixation reward.** In the Economic Choice runs the reward per step of
  correct fixation is 0.01 in the first stages and 0.001 once trials become
  long (from stage 5 in the full task and stage 8 in the partial one), so that
  the juice reward dominates.
* `EC_P_rnn_std` uses a learning rate of 0.001 and a longer stage 3 (10,000
  episodes instead of 5,000) than `EC_P_rnn`, hence 48,550 episodes in total.
* `CP_P_rnn` uses 100 hidden units and `gamma = 0.95`.
* `EC_P_ffnn` is trained with the shortened durations of the first curriculum
  stage only.
* **Noise.** The agent adds no noise to its observations (`noise_std = 0`). In
  the Economic Choice task the environment adds Gaussian noise of standard
  deviation `input_noise_sigma / sqrt(dt in s)` = 0.1 to the scaled drop counts
  from the second stage on.
* Seeds are set for NumPy, TensorFlow and Python, but the duration of the offer
  period is drawn from an unseeded generator, so retraining gives statistically
  similar, not identical, results.
