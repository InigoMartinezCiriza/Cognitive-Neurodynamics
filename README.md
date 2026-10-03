# Bioinspired RNN Models for Time-Dependent RL Tasks

Code, data and trained agents for the article *Bioinspired RNN Models for
Time-Dependent RL Tasks* (I. Martínez and C. M. Alaíz).

An actor–critic agent built on a bioinspired GRU (leaky, threshold-linear
units) is compared with a feedforward and a standard GRU network on CartPole
and on the economic choice task of Padoa-Schioppa & Assad (2006), including
partially observable variants of both.

## Contents

```
├── src/bioinspired_rnn/      models, training, environments and analysis
├── configs/                  one file per training run
├── data/                     training results and recorded activity
├── checkpoints/              trained agents
├── scripts/
│   ├── train.py              trains a run
│   ├── evaluate.py           runs a trained agent
│   ├── extract_data.py       raw training outputs -> data/
│   ├── record_activity.py    trained agents -> data/neural/
│   └── reproduce_paper.py    figures and tables, from data/
├── figures/                  generated figures
├── results/                  generated tables
└── tests/
```

## Installation

Python ≥ 3.11 (`requirements.txt` pins the versions used).

```bash
pip install -e .                 # analysis only
pip install -e ".[train,dev]"    # + training (TensorFlow, Keras 3, Gymnasium) and tests
```

## Method in brief

**Networks.** Actor and critic are `Dense(ReLU) → hidden layer → Dense`. The
hidden layer has 50 units (100 in the partial-CartPole modified RNN) and is
either a ReLU dense layer (FFNN), a Keras GRU (standard RNN) or the modified
GRU, with α = Δt/τ = 0.1:

```
λ_t = σ(W_λ,rec h_{t-1} + W_λ,in x_t + b_λ)          leak gate
γ_t = σ(W_γ,rec h_{t-1} + W_γ,in x_t + b_γ)          recurrent gate
h̃_t = W_rec (γ_t ⊙ h_{t-1}) + W_in x_t + b
h_t = ReLU(h_{t-1} + α λ_t ⊙ (h̃_t − h_{t-1}))
```

Only 10 % of the input and recurrent connections of the actor's hidden layer
are kept (fixed random mask); the critic is dense. The actor outputs action
probabilities; the critic receives the actor's hidden state and the one-hot
action and outputs the value baseline.

**Training.** REINFORCE with baseline, one update per episode, Adam (learning
rate 0.004; 0.001 for the standard RNN in the partial Economic Choice task),
L2 penalty 10⁻⁴, gradient clipping at global norm 1, no
discounting (γ = 1, except 0.95 for the partial-CartPole modified RNN). The
Economic Choice agents follow a curriculum that starts with 10–20 ms epochs and
lengthens them up to the original durations (fixation 1.5 s, offer 1–2 s,
decision ≤ 2 s, 10 ms steps); `configs/` lists every stage.

**Unit classification** (`analysis/neurons.py`). Each trained Economic Choice
RNN plays 500 trials of the original task with its weights fixed
(`scripts/record_activity.py`). The hidden state of every critic unit is
averaged over the last 100 steps of each completed trial (≈ the last second of
the offer period, since the agents choose within the first steps of the
decision period) and then over the trials of each *trial type* (offer and
juice chosen), as in Padoa-Schioppa & Assad (2006). The resulting tuning curve
is Min–Max normalised and regressed on offer value A (#A), offer value B (#B)
and chosen value (value of the juice chosen, in units of B). Units whose tuning
curve spans less than 0.001 are considered inactive; an active unit takes the
variable with the highest R² among those with a significant slope (p < 0.05),
and is unclassified otherwise.

## Training

```bash
python scripts/train.py EC_F_rnn                   # whole run
python scripts/train.py EC_F_rnn --stages 5-9      # resume from runs/Checkpoints/EC_F_rnn_4
python scripts/train.py CP_F_rnn --max-episodes 5  # quick test
```

Outputs go to `runs/Outputs/<run>_<stage>.pkl` (rewards, losses, hidden states
of every episode and trial records) and `runs/Checkpoints/<run>_<stage>/`, the
layout expected by `scripts/extract_data.py --raw-dir runs/Outputs`. Training
processes one time step at a time in eager mode, so the long Economic Choice
runs take many hours on a CPU. Reinforcement-learning training is stochastic:
retraining gives statistically similar, not identical, results.

The raw outputs of the paper's runs (about 6.5 GB) are not stored here; they
are available from the authors on request. `data/` contains everything the
analysis needs, and `data/sources.csv` lists the raw files it was extracted
from.

## Trained agents

`checkpoints/` holds the agent at the end of every run. For example,

```bash
python scripts/evaluate.py EC_F_rnn --episodes 500
```

plays 500 trials with the trained modified RNN and prints its psychometric
table. See `checkpoints/README.md` to load an agent in Python.

## Tests

```bash
pytest
```

Unit tests of the environments, the modified GRU equations and the analysis,
a short two-stage training run of each kind of agent (skipped without
TensorFlow), and the reproduction checks described above.

## Citation

If you use this code, please cite the article above; citation metadata for the
repository are in [`CITATION.cff`](CITATION.cff).

## Authors and license

Iñigo Martínez (imartinezc@estudiante.uam.es) and Carlos M. Alaíz
(carlos.alaiz@uam.es), Universidad Autónoma de Madrid.

Released under the [MIT License](LICENSE).

Released under the [MIT License](LICENSE).
