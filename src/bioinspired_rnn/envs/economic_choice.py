"""Economic choice task (Padoa-Schioppa & Assad, 2006) as a Gymnasium environment.

Each trial has three epochs:

* **Fixation** (``duration_params[0]`` ms): the fixation cue is on and the agent
  must fixate.
* **Offer** (uniform in ``[duration_params[1], duration_params[2]]`` ms): two
  juices, A and B, are offered on opposite sides with different numbers of
  drops; the agent must keep fixating.
* **Decision** (at most ``duration_params[3]`` ms): the fixation cue turns off
  and the agent chooses left or right. Choosing ends the trial and delivers the
  reward of the chosen side.

Breaking fixation before the decision epoch, or not choosing before the
decision timeout, aborts the trial with ``abort_penalty``. Fixating during the
decision epoch costs ``reward_go_fixation`` per step.

Observation (4 values): ``[fixation cue, juice position, drops left, drops
right]``. The juice position is -1 when juice A is on the left and +1 when it is
on the right; drop counts are divided by 10 and may carry Gaussian noise.

With ``partial=True`` (the *partial observables* variant of the paper) the offer
is only visible during the offer epoch: during the decision epoch the
observation is all zeros, so the agent must remember the offer.

Actions: 0 = fixate, 1 = choose left, 2 = choose right.
"""

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..task import OFFER_SETS

# Actions
ACT_FIXATE = 0
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# Observation indices
OBS_FIX_CUE = 0
OBS_JUICE_POS = 1
OBS_N_LEFT = 2
OBS_N_RIGHT = 3
OBS_DIM = 4

# Epochs
EPOCH_FIXATION = "Fixation"
EPOCH_OFFER = "Offer"
EPOCH_DECISION = "Decision"
EPOCH_END = "End"

#: Scaling applied to the number of drops in the observation.
DROPS_SCALE = 10.0


class EconomicChoiceEnv(gym.Env):
    """Economic choice task with fixation, offer and decision epochs.

    Args:
        dt: Simulation time step in ms.
        A_to_B_ratio: Value of one drop of A in drops of B.
        reward_B: Reward of one drop of B (one drop of A gives
            ``A_to_B_ratio * reward_B``).
        abort_penalty: Reward for breaking fixation or timing out.
        input_noise_sigma: Std of the Gaussian noise added to the (scaled) drop
            counts; it is multiplied by ``1/sqrt(dt in s)``.
        reward_fixation: Reward per step of correct fixation.
        reward_go_fixation: Reward per step of fixation during the decision epoch.
        duration_params: ``[fixation, offer_min, offer_max, decision_timeout]`` in ms.
        partial: Hide the offer during the decision epoch.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt=10, A_to_B_ratio=2.2, reward_B=100, abort_penalty=-0.1,
                 input_noise_sigma=0.0, reward_fixation=0.01, reward_go_fixation=-0.01,
                 duration_params=(1500, 1000, 2000, 2000), partial=False):
        super().__init__()

        if len(duration_params) != 4:
            raise ValueError("duration_params must be [fixation, offer_min, offer_max, "
                             "decision_timeout].")

        self.partial = partial
        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.sigma = input_noise_sigma
        self.noise_scale = 1.0 / np.sqrt(self.dt_sec) if self.dt_sec > 0 else 1.0
        self.R_fix_step = float(reward_fixation)
        self.R_go_fix_step = float(reward_go_fixation)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.1, high=2.1, shape=(OBS_DIM,), dtype=np.float32)

        self._durations_ms = {
            "Fixation": duration_params[0],
            "Offer_min": duration_params[1],
            "Offer_max": duration_params[2],
            "Decision_timeout": duration_params[3],
        }
        self.t_fixation_steps = self._ms_to_steps(self._durations_ms["Fixation"])
        self.t_choice_timeout_steps = self._ms_to_steps(self._durations_ms["Decision_timeout"])

        self.juice_types = [("A", "B"), ("B", "A")]
        self.offer_sets = list(OFFER_SETS)
        self.rng = np.random.default_rng()

        # Trial state
        self.current_step = 0
        self.trial_juice_LR = None   # ('A', 'B') or ('B', 'A')
        self.trial_offer_BA = None   # (n_B, n_A)
        self.trial_nL = 0
        self.trial_nR = 0
        self.trial_rL = 0.0
        self.trial_rR = 0.0
        self.epochs = {}
        self.current_epoch_name = EPOCH_END
        self.t_go_signal_step = -1
        self.t_choice_made_step = -1
        self.chosen_action = -1

    # ------------------------------------------------------------------ timing
    def _ms_to_steps(self, ms):
        """Milliseconds to steps (at least one step for any positive duration)."""
        return max(1, int(np.round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, delay_ms):
        t_fix_end = self.t_fixation_steps
        t_go_signal = t_fix_end + self._ms_to_steps(delay_ms)
        t_choice_end = t_go_signal + self.t_choice_timeout_steps
        t_max = t_choice_end + self._ms_to_steps(100)
        self.epochs = {
            EPOCH_FIXATION: (0, t_fix_end),
            EPOCH_OFFER: (t_fix_end, t_go_signal),
            EPOCH_DECISION: (t_go_signal, t_choice_end),
            EPOCH_END: (t_max, t_max + 1),
            "tmax_steps": t_max,
        }
        self.t_go_signal_step = t_go_signal

    def _get_current_epoch(self, step):
        if self.epochs[EPOCH_FIXATION][0] <= step < self.epochs[EPOCH_FIXATION][1]:
            return EPOCH_FIXATION
        if self.epochs[EPOCH_OFFER][0] <= step < self.epochs[EPOCH_OFFER][1]:
            return EPOCH_OFFER
        if self.epochs[EPOCH_DECISION][0] <= step < self.epochs[EPOCH_DECISION][1]:
            return EPOCH_DECISION if self.t_choice_made_step == -1 else EPOCH_END
        return EPOCH_END

    # ------------------------------------------------------------------ trials
    def _select_trial_conditions(self):
        """Draws juice sides, offer and offer duration for a new trial."""
        self.trial_juice_LR = random.choice(self.juice_types)
        nB, nA = random.choice(self.offer_sets)
        self.trial_offer_BA = (nB, nA)

        if self.trial_juice_LR[0] == "A":
            self.trial_nL, self.trial_nR = nA, nB
            self.trial_rL, self.trial_rR = nA * self.R_A, nB * self.R_B
        else:
            self.trial_nL, self.trial_nR = nB, nA
            self.trial_rL, self.trial_rR = nB * self.R_B, nA * self.R_A

        delay_ms = self.rng.uniform(self._durations_ms["Offer_min"], self._durations_ms["Offer_max"])
        self._calculate_epochs(delay_ms)

    def _get_observation(self, current_epoch):
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if current_epoch in (EPOCH_FIXATION, EPOCH_OFFER):
            obs[OBS_FIX_CUE] = 1.0

        offer_visible = current_epoch == EPOCH_OFFER or (
            current_epoch == EPOCH_DECISION and not self.partial)
        if offer_visible:
            obs[OBS_JUICE_POS] = -1.0 if self.trial_juice_LR[0] == "A" else 1.0
            scaled_nL = self.trial_nL / DROPS_SCALE
            scaled_nR = self.trial_nR / DROPS_SCALE
            if self.sigma > 0:
                scaled_nL += self.rng.normal(scale=self.sigma) * self.noise_scale
                scaled_nR += self.rng.normal(scale=self.sigma) * self.noise_scale
            obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, 1.1)
            obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, 1.1)
            obs[OBS_JUICE_POS] = np.clip(obs[OBS_JUICE_POS], -1.0, 1.0)

        if current_epoch == EPOCH_END or (current_epoch == EPOCH_DECISION and self.partial):
            return np.zeros(OBS_DIM, dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # --------------------------------------------------------------- Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self.current_step = 0
        self._select_trial_conditions()
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        self.t_choice_made_step = -1
        self.chosen_action = -1

        info = self._get_info()
        info["reward"] = 0.0
        info["action"] = None
        return self._get_observation(self.current_epoch_name), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        terminated = False
        truncated = False
        reward = 0.0
        prev_epoch = self.current_epoch_name

        if prev_epoch in (EPOCH_FIXATION, EPOCH_OFFER):
            if action != ACT_FIXATE:            # broke fixation
                reward = self.R_ABORTED
                terminated = True
                self.current_epoch_name = EPOCH_END
            else:
                reward = self.R_fix_step
        elif prev_epoch == EPOCH_DECISION:
            if action == ACT_FIXATE:
                reward = self.R_go_fix_step
            else:                               # a choice ends the trial
                self.t_choice_made_step = self.current_step
                self.chosen_action = action
                terminated = True
                reward = self.trial_rL if action == ACT_CHOOSE_LEFT else self.trial_rR
                self.current_epoch_name = EPOCH_END

        if not terminated:
            self.current_step += 1
            next_epoch = self._get_current_epoch(self.current_step)
            go_end = self.epochs[EPOCH_DECISION][1]
            if (prev_epoch == EPOCH_DECISION and self.current_step >= go_end
                    and self.t_choice_made_step == -1):
                reward = self.R_ABORTED             # decision timeout
                terminated = True
                next_epoch = EPOCH_END
            elif self.current_step >= self.epochs["tmax_steps"]:
                truncated = True
                reward = 0.0
                next_epoch = EPOCH_END
            self.current_epoch_name = next_epoch

        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["action"] = action
        info["reward"] = reward
        if terminated:
            truncated = False
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        is_correct = None
        if self.chosen_action == ACT_CHOOSE_LEFT:
            is_correct = self.trial_rL >= self.trial_rR
        elif self.chosen_action == ACT_CHOOSE_RIGHT:
            is_correct = self.trial_rR >= self.trial_rL

        return {
            "step": self.current_step,
            "epoch": self.current_epoch_name,
            "juice_LR": self.trial_juice_LR,
            "offer_BA": self.trial_offer_BA,
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "rL": self.trial_rL,
            "rR": self.trial_rR,
            "chosen_action": self.chosen_action,
            "choice_time_step": self.t_choice_made_step,
            "is_correct_choice": is_correct,
            "A_to_B_ratio": self.A_to_B_ratio,
            "rewards_cfg": {
                "fix_step": self.R_fix_step,
                "go_fix_step": self.R_go_fix_step,
                "abort": self.R_ABORTED,
            },
        }

    def render(self):
        pass

    def close(self):
        pass
