"""Reproduces the figures and the quantitative results of the paper from ``data/``.

Writes:

* ``figures/fig3a_cartpole.pdf``, ``figures/fig3b_cartpole_partial.pdf``
* ``figures/fig4a_psychometric_full.pdf``, ``figures/fig4b_psychometric_partial.pdf``
* ``figures/fig5_representative_neurons.pdf``
  (and a PNG copy of each figure)
* ``results/table1_neuron_types.csv``: Table 1
* ``results/rank_sum_offer_A_vs_B.csv``: Wilcoxon rank-sum test of Section 4.2
* ``results/neuron_classification.csv``: regressions and type of every unit
* ``results/fig5_exemplars.csv``: units shown in Fig. 5
* ``results/psychometric.csv``: data of Fig. 4
* ``results/training_summary.csv``: episodes per run and learning speed (Section 4)

Usage::

    python scripts/reproduce_paper.py
"""

import argparse
from pathlib import Path

import pandas as pd

from bioinspired_rnn.analysis import behaviour, neurons, plots
from bioinspired_rnn.analysis.data import (DATA_DIR, NEURAL_RUNS, REPO_ROOT, all_configs,
                                           load_curves, load_neural, load_trials)
from bioinspired_rnn.task import ENVIRONMENTS, NETWORKS

#: Moving-average windows of Fig. 3 (episodes).
WINDOWS = {
    "F": {"FFNN": 70, "Modified RNN": 70, "Standard RNN": 70},
    "P": {"FFNN": 800, "Modified RNN": 800, "Standard RNN": 100},
}
#: Fig. 3b: the standard RNN ran 1000 episodes and is stretched to the 8000 of the others.
SCALE_PARTIAL = {"Standard RNN": 8}
#: Order (and therefore colour) of the networks in Figs. 3 and 4.
PLOT_ORDER = ["ffnn", "rnn", "rnn_std"]
#: Reward that counts as "learnt" in the learning-speed summary.
LEARNT_REWARD = 200


def save_figure(fig_fn, path, *args, **kwargs):
    fig = fig_fn(*args, **kwargs)
    # No creation date, so that regenerated PDFs are byte-identical
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plots.plt.close(fig)
    print(f"  {path.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def figure3(fig_dir):
    for env, name in (("F", "fig3a_cartpole"), ("P", "fig3b_cartpole_partial")):
        rewards = {NETWORKS[net]: load_curves(f"CP_{env}_{net}")["total_reward"]
                   for net in PLOT_ORDER}
        save_figure(plots.learning_curves, fig_dir / name, rewards, WINDOWS[env],
                    scale=SCALE_PARTIAL if env == "P" else None)


def figure4(fig_dir, results_dir):
    rows = []
    for env, name in (("F", "fig4a_psychometric_full"), ("P", "fig4b_psychometric_partial")):
        tables = {}
        for net in PLOT_ORDER:
            run = f"EC_{env}_{net}"
            trials = load_trials(run, stages="psychometric")
            tables[NETWORKS[net]] = behaviour.psychometric(trials)
            rows.append(tables[NETWORKS[net]].assign(
                environment=ENVIRONMENTS[env], network=NETWORKS[net], run=run,
                stages=" ".join(map(str, sorted(trials["stage"].unique())))))
        save_figure(plots.psychometric_curves, fig_dir / name, tables)
    columns = ["environment", "network", "run", "stages", "offer", "n_B", "n_A", "trials",
               "percent_B"]
    pd.concat(rows)[columns].to_csv(results_dir / "psychometric.csv", index=False,
                                    lineterminator="\n")


def neuron_analysis(fig_dir, results_dir):
    analyses = {}
    for key, run in NEURAL_RUNS.items():
        trials, activity = load_neural(run)
        analyses[key] = neurons.analyse_units(activity, trials)

    # Per-unit classification
    per_unit = pd.concat(
        [a.fits.reset_index().assign(environment=env, network=net, run=NEURAL_RUNS[(env, net)])
         for (env, net), a in analyses.items()], ignore_index=True)
    first = ["environment", "network", "run", "unit", "type", "modulation"]
    per_unit = per_unit[first + [c for c in per_unit.columns if c not in first]]
    per_unit.to_csv(results_dir / "neuron_classification.csv", index=False, lineterminator="\n")

    # Table 1
    table1 = neurons.type_counts(analyses)
    table1.to_csv(results_dir / "table1_neuron_types.csv", lineterminator="\n")
    print("\nTable 1 - number of critic units of each type:")
    print(table1.to_string())

    # Wilcoxon rank-sum test, Offer A vs Offer B |slope|
    tests = pd.DataFrame([{"environment": env, "network": net,
                           **neurons.offer_slopes_rank_sum(a.fits)}
                          for (env, net), a in analyses.items()])
    tests.to_csv(results_dir / "rank_sum_offer_A_vs_B.csv", index=False, lineterminator="\n")
    print("\nWilcoxon rank-sum test, |slope| of Offer A vs Offer B units:")
    print(tests.to_string(index=False))

    # Fig. 5
    exemplars, rows = {}, []
    for (env, net), a in analyses.items():
        for var in neurons.VARIABLES:
            ex = neurons.exemplar(a, var)
            exemplars[(env, net, var)] = None if ex is None else ex["unit"]
            if ex is not None:
                rows.append({"environment": env, "network": net, "variable": var, **ex})
    pd.DataFrame(rows).to_csv(results_dir / "fig5_exemplars.csv", index=False,
                              lineterminator="\n")
    save_figure(plots.representative_neurons, fig_dir / "fig5_representative_neurons",
                analyses, exemplars)


def training_summary(results_dir):
    rows = []
    for name, config in all_configs().items():
        curves = load_curves(name)
        task, env, net = name.split("_", 2)
        row = {"run": name, "task": "CartPole" if task == "CP" else "Economic choice",
               "environment": ENVIRONMENTS[env], "network": NETWORKS[net],
               "stages": config.n_stages, "episodes": len(curves),
               "mean_reward_last_100": curves["total_reward"].tail(100).mean()}
        if task == "CP":
            window = WINDOWS[env][NETWORKS[net]]
            first = behaviour.first_episode_above(curves["total_reward"], LEARNT_REWARD, window)
            row[f"first_episode_moving_avg_ge_{LEARNT_REWARD}"] = first
            row["moving_window"] = window
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "training_summary.csv", index=False, lineterminator="\n")
    print("\nTraining summary:")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()
    fig_dir, results_dir = Path(args.figures_dir), Path(args.results_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading processed data from {DATA_DIR}")
    print("Figures:")
    figure3(fig_dir)
    figure4(fig_dir, results_dir)
    neuron_analysis(fig_dir, results_dir)
    training_summary(results_dir)


if __name__ == "__main__":
    main()
