import numpy as np
import gymnasium as gym
from gymnasium import spaces

import simulator as gm


G_MIN = 150
G_MAX = 255
N_G = G_MAX - G_MIN + 1

N_LAYERS = 6
LAYER_THICKNESS_UM = 50.0
BETA_TIME = 1.0


def required_Cd_for_layer(j, layer_thickness_um=LAYER_THICKNESS_UM):
    if j == 1:
        return layer_thickness_um
    return 2.0 * layer_thickness_um


class RoofEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_layers=N_LAYERS,
        layer_thickness_um=LAYER_THICKNESS_UM,
        I_OI=gm.I_OI_DEFAULT,
        Omega_g=gm.OMEGA_G_DEFAULT,
        beta_time=BETA_TIME,
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

    @staticmethod
    def _obs():
        return np.zeros(1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).flatten()
        if action.shape != (self.n_layers,):
            raise ValueError(f"Expected action shape {(self.n_layers,)}, got {action.shape}.")

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
        z_overcure = float(result["final_overcure_void_depth_um"])
        z_gel_only = float(result["final_gel_only_band_depth_um"])
        t_total = float(sum(t_history))

        reward = -(z_gel + z_overcure) - self.beta_time * t_total

        info = {
            "g_history": g_history,
            "t_history": t_history,
            "Cd_required_um": Cd_required_um,
            "final_gel_boundary_um": float(result["final_gel_boundary_um"]),
            "final_overcure_boundary_um": float(result["final_overcure_boundary_um"]),
            "final_gel_void_depth_um": z_gel,
            "final_overcure_void_depth_um": z_overcure,
            "final_gel_only_band_depth_um": z_gel_only,
            "total_exposure_time_s": t_total,
            "per_layer": result["per_layer"],
        }

        terminated = True
        truncated = False
        return self._obs(), float(reward), terminated, truncated, info
