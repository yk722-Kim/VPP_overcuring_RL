"""
Step-based gym environment for the progressively widening structure.

Episode:
    16 sequential steps, one per layer.
    At step j (1..16), the agent picks g_j; t_j is determined by
    Cd = 50 um equality constraint  ->  t_j = Tc(g_j) * exp(50/Dp(g_j)).

Per-layer effect:
    Layer j's exposure is delivered to all overhangs i = 1..j.
    Each overhang i is a 1D OverhangColumn that gets one more
    expose_one_layer(g_j, t_j) call. Overhang j is created at step j.

Observation (17-dim):
    s[0..15]: for each overhang i = 1..16,
              (gel_only_depth_i + solid_depth_i) / 50  (clipped to [0,1])
              0 if overhang i has not been created yet (i > j).
    s[16]:    j / 16  (step progress)

Action (Discrete 106):
    g index 0..105 mapped to g = 150..255.

Reward (dense):
    Negative incremental overcure growth caused by this exposure,
    summed over all currently existing overhangs:
        r_j = -sum_{i=1..j} delta_overcure_i
    where delta_overcure_i = (gel_only + solid)_after_j - (gel_only + solid)_before_j
    For overhang j (newly created) the "before" is 0 by convention.
    Sum of r_j over the episode equals
        -sum_i (final gel_only_i + final solid_i)
    so the dense and sparse rewards are equivalent in total return.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import (
    OverhangColumn,
    t_for_required_Cd,
    I_OI_DEFAULT, OMEGA_G_DEFAULT,
)


# ---- Action space ----
G_MIN = 150
G_MAX = 255
N_G = G_MAX - G_MIN + 1   # 106

# ---- Geometry ----
N_LAYERS = 16
LAYER_THICKNESS_UM = 50.0
REQUIRED_CD_UM = 50.0     # all layers cure exactly 50 um
OBS_NORM_UM = 50.0        # state normalization scale


class WideningEnv(gym.Env):
    """One step per layer; one episode per 16-layer build."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_layers=N_LAYERS,
        layer_thickness_um=LAYER_THICKNESS_UM,
        required_Cd_um=REQUIRED_CD_UM,
        I_OI=I_OI_DEFAULT,
        Omega_g=OMEGA_G_DEFAULT,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_thickness_um = layer_thickness_um
        self.required_Cd_um = required_Cd_um
        self.I_OI = I_OI
        self.Omega_g = Omega_g

        self.action_space = spaces.Discrete(N_G)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_layers + 1,), dtype=np.float32
        )

        self._reset_internal()

    # ------------------------------------------------------------------
    def _reset_internal(self):
        self.current_step = 0    # increments to 1, 2, ... up to n_layers
        self.columns = []        # OverhangColumn instances, len grows over steps
        self.last_overcure_per_overhang = []   # snapshot before current step
        self.g_history = []
        self.t_history = []

    def _obs(self):
        s = np.zeros(self.n_layers + 1, dtype=np.float32)
        for i, col in enumerate(self.columns):
            depth = col.gel_only_void_depth_um + col.solid_void_depth_um
            s[i] = float(np.clip(depth / OBS_NORM_UM, 0.0, 1.0))
        s[self.n_layers] = float(self.current_step) / self.n_layers
        return s

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._obs(), {}

    def step(self, action):
        assert self.current_step < self.n_layers, "Episode already terminated"

        a = int(action)
        g = a + G_MIN
        t = t_for_required_Cd(g, self.required_Cd_um)
        self.current_step += 1
        j = self.current_step

        self.g_history.append(g)
        self.t_history.append(t)

        # Snapshot existing overhangs' overcure before this exposure
        before = [
            col.gel_only_void_depth_um + col.solid_void_depth_um
            for col in self.columns
        ]

        # Create overhang j
        new_col = OverhangColumn(
            layer_thickness_um=self.layer_thickness_um,
            I_OI=self.I_OI,
            Omega_g=self.Omega_g,
        )
        self.columns.append(new_col)

        # Apply the exposure to every existing overhang (including the new one)
        for col in self.columns:
            col.expose_one_layer(g, t)

        # After-state
        after = [
            col.gel_only_void_depth_um + col.solid_void_depth_um
            for col in self.columns
        ]

        # Reward = -(sum of incremental overcure growth this step)
        # before is shorter than after by 1 (the new overhang), pad with 0.
        before_padded = before + [0.0]
        delta = [a_v - b_v for a_v, b_v in zip(after, before_padded)]
        reward = -float(sum(delta)) - 1.0 * t

        terminated = (self.current_step >= self.n_layers)
        truncated = False

        if terminated:
            total_overcure = sum(after)
            info = {
                "g_history": list(self.g_history),
                "t_history": list(self.t_history),
                "per_overhang_solid_um": [c.solid_void_depth_um for c in self.columns],
                "per_overhang_gel_only_um": [c.gel_only_void_depth_um for c in self.columns],
                "per_overhang_total_um": after,
                "total_overcure_um": total_overcure,
            }
        else:
            info = {
                "step_delta_um": delta,
                "step_reward": reward,
            }

        return self._obs(), reward, terminated, truncated, info


# ---- Sanity: drive the env with conventional g=255 across all 16 layers ----
if __name__ == "__main__":
    env = WideningEnv()
    obs, _ = env.reset()

    cumulative_reward = 0.0
    for step in range(N_LAYERS):
        action = 255 - G_MIN   # g=255 always
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        if not terminated:
            print(f"step {step+1}: g=255, reward={reward:.3f}, "
                  f"step_delta={['%.2f' % d for d in info['step_delta_um']]}")
        else:
            print(f"\nstep {step+1} (terminal): g=255, reward={reward:.3f}")
            print(f"\nFinal per-overhang totals (gel_only + solid):")
            for i, v in enumerate(info["per_overhang_total_um"]):
                print(f"  overhang {i+1:>2}: solid={info['per_overhang_solid_um'][i]:>7.2f} um, "
                      f"gel_only={info['per_overhang_gel_only_um'][i]:>7.2f} um, "
                      f"total={v:>7.2f} um")
            print(f"\nTotal overcure = {info['total_overcure_um']:.3f} um")
            print(f"Total return   = {cumulative_reward:.3f}")
