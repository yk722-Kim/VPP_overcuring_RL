import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import (
    OverhangColumn,
    t_for_required_Cd,
    I_OI_DEFAULT,
    OMEGA_G_DEFAULT,
)

G_MIN = 150
G_MAX = 255
N_G = G_MAX - G_MIN + 1

N_LAYERS = 16
LAYER_THICKNESS_UM = 50.0
REQUIRED_CD_UM = 50.0
OBS_NORM_UM = 50.0
BETA_TIME = 1.0


class SlantEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_layers=N_LAYERS,
        layer_thickness_um=LAYER_THICKNESS_UM,
        required_Cd_um=REQUIRED_CD_UM,
        I_OI=I_OI_DEFAULT,
        Omega_g=OMEGA_G_DEFAULT,
        beta_time=BETA_TIME,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layer_thickness_um = float(layer_thickness_um)
        self.required_Cd_um = float(required_Cd_um)
        self.I_OI = float(I_OI)
        self.Omega_g = float(Omega_g)
        self.beta_time = float(beta_time)

        self.action_space = spaces.Discrete(N_G)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self._reset_internal()

    def _reset_internal(self):
        self.current_step = 0
        self.finished_columns = []
        self.current_pending = None
        self.g_history = []
        self.t_history = []

    @staticmethod
    def _objective_depth(col):
        z_gel = float(col.gel_void_depth_um)
        z_overcure = float(col.solid_void_depth_um)
        return z_gel + z_overcure

    def _obs(self):
        obs = np.zeros(3, dtype=np.float32)
        obs[0] = float(self.current_step) / self.n_layers

        if self.current_pending is not None:
            obs[1] = float(
                np.clip(self.current_pending.gel_void_depth_um / OBS_NORM_UM, 0.0, 1.0)
            )

        if self.finished_columns:
            obs[2] = float(
                np.clip(self.finished_columns[-1].gel_void_depth_um / OBS_NORM_UM, 0.0, 1.0)
            )

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._obs(), {}

    def step(self, action):
        if self.current_step >= self.n_layers:
            raise RuntimeError("Episode already terminated. Call reset() before step().")

        action_index = int(action)
        if action_index < 0 or action_index >= N_G:
            raise ValueError(f"Action must be in [0, {N_G - 1}], got {action_index}.")

        g = action_index + G_MIN
        t = t_for_required_Cd(g, self.required_Cd_um)

        self.current_step += 1
        self.g_history.append(g)
        self.t_history.append(t)

        delta_objective = 0.0

        if self.current_pending is not None:
            before = self._objective_depth(self.current_pending)
            self.current_pending.expose_one_layer(g, t)
            after = self._objective_depth(self.current_pending)
            delta_objective += after - before
            self.finished_columns.append(self.current_pending)
            self.current_pending = None

        new_col = OverhangColumn(
            layer_thickness_um=self.layer_thickness_um,
            I_OI=self.I_OI,
            Omega_g=self.Omega_g,
        )
        new_col.expose_one_layer(g, t)
        delta_objective += self._objective_depth(new_col)
        self.current_pending = new_col

        reward = -float(delta_objective) - self.beta_time * float(t)

        terminated = self.current_step >= self.n_layers
        truncated = False

        info = {
            "g_history": list(self.g_history),
            "t_history": list(self.t_history),
            "step_delta_objective_um": float(delta_objective),
        }

        if terminated:
            if self.current_pending is not None:
                self.finished_columns.append(self.current_pending)
                self.current_pending = None

            if len(self.finished_columns) != self.n_layers:
                raise RuntimeError(
                    f"Expected {self.n_layers} columns, got {len(self.finished_columns)}."
                )

            per_overhang_z_gel = [float(c.gel_void_depth_um) for c in self.finished_columns]
            per_overhang_z_overcure = [
                float(c.solid_void_depth_um) for c in self.finished_columns
            ]
            per_overhang_z_gel_only = [
                float(c.gel_only_void_depth_um) for c in self.finished_columns
            ]
            per_overhang_objective = [
                z_gel + z_overcure
                for z_gel, z_overcure in zip(
                    per_overhang_z_gel,
                    per_overhang_z_overcure,
                )
            ]

            info.update(
                {
                    "per_overhang_z_gel_um": per_overhang_z_gel,
                    "per_overhang_z_overcure_um": per_overhang_z_overcure,
                    "per_overhang_z_gel_only_um": per_overhang_z_gel_only,
                    "per_overhang_objective_um": per_overhang_objective,
                    "total_z_gel_um": float(sum(per_overhang_z_gel)),
                    "total_z_overcure_um": float(sum(per_overhang_z_overcure)),
                    "total_objective_um": float(sum(per_overhang_objective)),
                    "total_exposure_time_s": float(sum(self.t_history)),
                }
            )

        return self._obs(), float(reward), terminated, truncated, info
