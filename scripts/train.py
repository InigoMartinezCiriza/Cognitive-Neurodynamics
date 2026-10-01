"""Trains one run (all its curriculum stages, or a subset).

Examples::

    # Full run, outputs in runs/Outputs and runs/Checkpoints
    python scripts/train.py EC_F_rnn

    # Resume from stage 5 (needs runs/Checkpoints/EC_F_rnn_4)
    python scripts/train.py EC_F_rnn --stages 5-9

    # Quick smoke test: at most 5 episodes per stage
    python scripts/train.py configs/CP_F_rnn.toml --max-episodes 5 --output-dir runs/smoke
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from bioinspired_rnn.config import load_config  # noqa: E402
from bioinspired_rnn.training import run  # noqa: E402


def parse_stages(text, n_stages):
    if text is None:
        return list(range(1, n_stages + 1))
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="run name (e.g. EC_F_rnn) or path to a .toml file")
    parser.add_argument("--output-dir", default="runs", help="where Outputs/ and Checkpoints/ go")
    parser.add_argument("--stages", help="stage or range of stages, e.g. 3 or 3-9 (default: all)")
    parser.add_argument("--max-episodes", type=int, help="cap the episodes of each stage")
    parser.add_argument("--print-interval", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, args.output_dir, stages=parse_stages(args.stages, config.n_stages),
        max_episodes=args.max_episodes, print_interval=args.print_interval)


if __name__ == "__main__":
    main()
