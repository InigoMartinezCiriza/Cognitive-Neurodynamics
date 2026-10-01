# Processed data

Everything the figures and tables of the paper need. The learning curves and
trial records are extracted from the raw training outputs with

```bash
python scripts/extract_data.py --raw-dir path/to/Outputs
```

and the neural activity is recorded from the trained agents in `checkpoints/`
with

```bash
python scripts/record_activity.py
```

The raw outputs (`<run>_<stage>.pkl`, one per curriculum stage, about 6.5 GB in
total because they store the hidden states of every episode) are not included
in the repository; `sources.csv` lists the files used. Runs are named
`<task>_<environment>_<network>`:

| Part | Values |
|---|---|
| task | `CP` CartPole, `EC` Economic Choice |
| environment | `F` full observables, `P` partial observables |
| network | `ffnn` FFNN, `rnn_std` standard GRU RNN, `rnn` modified (bioinspired) GRU RNN |

## `training_curves/<run>.csv`

One row per training episode, all stages of the run concatenated.

| Column | Description |
|---|---|
| `stage` | Curriculum stage (see `configs/<run>.toml`) |
| `episode` | Episode number within the run, from 1 |
| `total_reward` | Sum of the rewards of the episode |
| `actor_loss`, `critic_loss` | Losses of the update made after the episode (including the L2 term) |

## `trials/<run>.csv` (Economic Choice runs)

One row per training episode (trial).

| Column | Description |
|---|---|
| `stage`, `episode` | As above |
| `completed` | `True` if the trial ended with a left/right choice, `False` if it was aborted (fixation break or timeout) |
| `juice_left`, `juice_right` | Juice shown on each side (`A` or `B`) |
| `n_B`, `n_A` | Offer: drops of juice B and of juice A |
| `chosen_juice` | Juice chosen (`A` or `B`) |

The offer and choice are only recorded for completed trials.

## `neural/<run>_{critic,actor}_offer_window.csv` (Economic Choice RNNs)

Activity of the hidden units of the critic (value network, analysed in the
paper) and of the actor, recorded while the trained agent (`checkpoints/`)
plays 500 trials of the original task (fixation 1.5 s, offer 1-2 s,
decision ≤ 2 s, 10 ms per step) without learning. Both networks run with their
recurrent dynamics over the whole trial. One row per completed trial.

| Column | Description |
|---|---|
| `trial` | Trial number (1-500) |
| `n_B`, `n_A`, `juice_left`, `chosen_juice` | Offer and choice of the trial |
| `n_steps` | Number of steps of the trial |
| `unit_00` ... `unit_49` | Hidden state of each unit averaged over the last 100 steps of the trial |

Since the agents choose within the first steps of the decision period, the last
100 steps are, in practice, the last second of the offer period. The values are
written with full float64 precision; read them with
`pandas.read_csv(..., float_precision="round_trip")` (as
`bioinspired_rnn.analysis.data.load_neural` does) to recover them exactly.

## `sources.csv`

Raw file of each stage, its size in bytes and number of episodes.
