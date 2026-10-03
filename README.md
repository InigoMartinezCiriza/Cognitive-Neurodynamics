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

## Usage

```bash
python scripts/train.py EC_F_rnn                     # train a run
python scripts/evaluate.py EC_F_rnn --episodes 500   # run a trained agent
pytest                                               # tests
```

The raw training outputs (about 6.5 GB) are available from the authors on
request. On Windows, TensorFlow cannot read checkpoints from paths with
non-ASCII characters.

## Citation

Please cite the article above; citation metadata are in
[`CITATION.cff`](CITATION.cff).

## Authors and license

Iñigo Martínez (imartinezc@estudiante.uam.es) and Carlos M. Alaíz
(carlos.alaiz@uam.es), Universidad Autónoma de Madrid.

Released under the [MIT License](LICENSE).
