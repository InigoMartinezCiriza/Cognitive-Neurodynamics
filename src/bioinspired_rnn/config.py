"""Experiment configurations (``configs/*.toml``).

Each file describes one *run*: an agent trained on one environment through a
sequence of curriculum *stages*. A stage is a block of episodes with fixed
environment settings; the agent of stage ``n`` starts from the checkpoint saved
at the end of stage ``n - 1``. See ``configs/README.md`` for the format.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class RunConfig:
    name: str
    task: str                      # "cartpole" or "economic_choice"
    partial: bool
    seed: int
    agent: dict
    training: dict
    env: dict
    stages: list = field(default_factory=list)
    description: str = ""

    @property
    def n_stages(self):
        return len(self.stages)

    @property
    def episodes_per_stage(self):
        return [stage["episodes"] for stage in self.stages]

    @property
    def total_episodes(self):
        return sum(self.episodes_per_stage)

    def stage_env_params(self, stage):
        """Environment parameters of stage ``stage`` (1-based)."""
        params = dict(self.env)
        params.update({k: v for k, v in self.stages[stage - 1].items()
                       if k not in ("episodes", "learning_rate")})
        return params


def load_config(path_or_name):
    """Loads a run configuration from a path or from its name (e.g. ``"EC_F_rnn"``)."""
    path = Path(path_or_name)
    if not path.suffix:
        path = CONFIG_DIR / f"{path_or_name}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return RunConfig(
        name=raw["name"],
        task=raw["task"],
        partial=raw.get("partial", False),
        seed=raw.get("seed", 1),
        agent=raw["agent"],
        training=raw["training"],
        env=raw.get("env", {}),
        stages=raw["stages"],
        description=raw.get("description", ""),
    )


def all_configs():
    """All run configurations in ``configs/``, keyed by run name."""
    return {cfg.name: cfg for cfg in map(load_config, sorted(CONFIG_DIR.glob("*.toml")))}
