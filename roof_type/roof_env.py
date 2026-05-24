"""
Gymnasium environment for the roof-type overhang VPP model.

Roof-type formulation:
  - One episode corresponds to one complete 6-layer roof-type overhang build.
  - The action is a full grayscale schedule, one grayscale value per layer.
  - Each grayscale value g_j is selected from {150, 151, ..., 255}.
  - The exposure time t_j is computed from the required cure depth using
    t_for_required_Cd(g_j, Cd_req_j).
  - Cd_req_1 = 50 um and Cd_req_j = 100 um for j = 2, ..., 6.

Reward:
  R = -(z_gel + z_overcure) - beta * sum_j(t_j)

where:
  - z_gel is the final gel-memory void depth beyond the nominal boundary.
  - z_overcure is the final fully solidified void depth beyond the nominal boundary.
  - beta is the exposure-time penalty coefficient.

The simulator key "final_solid_void_depth_um" corresponds to z_overcure in the
manuscript. The simulator key "final_gel_void_depth_um" corresponds to z_gel.

Observation:
  This is a contextless bandit-style environment for the roof-type case, so the
  observation is a dummy scalar required by Stable-Baselines3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import simulator as gm


# ---- Action space ----
G_MIN = 150
G_MAX = 255
N_G = G_MAX - G_MIN + 1

# ---- Geometry ----
N_LAYERS = 6
LAYER_THICKNESS_UM = 50.0

# ---- Reward ----
DEFAULT_BETA_TIME = 1.0


def required_Cd_for_layer(j, layer_thickness_um=LAYER_THICKNESS_UM):
    """Return the required cure depth for a 1-indexed roof-type layer."""
    if j == 1:
        return layer_thickness_um
    return 2.0 * layer_thickness_um


class RoofOverhangGelEnv(gym.Env):
    """Roof-type overhang environment with one full grayscale schedule per episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_layers=N_LAYERS,
        layer_thickness_um=LAYER_THICKNESS_UM,
        I_OI=gm.I_OI_DEFAULT,
        Omega_g=gm.OMEGA_G_DEFAULT,
        beta_time=DEFAULT_BETA_TIME,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layer_thickness_um = float(layer_thickness_um)
        self.I_OI = float(I_OI)
        self.Omega_g = float(Omega_g)
        self.beta_time = float(beta_time)

        self.action_space = spaces.MultiDiscrete([N_G] * self.n_layers)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._last_info = {}

    @staticmethod
    def _obs():
        return np.zeros(1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._last_info = {}
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).flatten()
        if action.shape != (self.n_layers,):
            raise ValueError(
                f"Expected action shape {(self.n_layers,)}, got {action.shape}."
            )

        g_history = [int(a) + G_MIN for a in action]
        Cd_required_um = [
            required_Cd_for_layer(j + 1, self.layer_thickness_um)
            for j in range(self.n_layers)
        ]
        t_history = [
            gm.t_for_required_Cd(g, Cd_req)
            for g, Cd_req in zip(g_history, Cd_required_um)
        ]

        result = gm.simulate_gel_memory(
            g_history,
            t_history,
            I_OI=self.I_OI,
            Omega_g=self.Omega_g,
            layer_thickness_um=self.layer_thickness_um,
        )

        z_gel = float(result["final_gel_void_depth_um"])
        z_overcure = float(result["final_solid_void_depth_um"])
        z_gel_only = float(result["final_gel_only_void_depth_um"])
        t_total = float(sum(t_history))

        reward_overcure_part = -(z_gel + z_overcure)
        reward_time_part = -self.beta_time * t_total
        reward = reward_overcure_part + reward_time_part

        info = {
            "g_history": g_history,
            "t_history": t_history,
            "Cd_required_um": Cd_required_um,
            "final_gel_boundary_um": float(result["final_gel_boundary_um"]),
            "final_solid_boundary_um": float(result["final_solid_boundary_um"]),
            "final_gel_void_depth_um": z_gel,
            "final_solid_void_depth_um": z_overcure,
            "final_gel_only_void_depth_um": z_gel_only,
            "z_gel_um": z_gel,
            "z_overcure_um": z_overcure,
            "z_gel_only_um": z_gel_only,
            "t_total": t_total,
            "beta_time": self.beta_time,
            "reward_overcure_part": reward_overcure_part,
            "reward_time_part": reward_time_part,
            "per_layer": result["per_layer"],
        }

        self._last_info = info
        terminated = True
        truncated = False
        return self._obs(), float(reward), terminated, truncated, info
