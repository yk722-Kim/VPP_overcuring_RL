import numpy as np


def I0_from_g(g):
    g = np.asarray(g, dtype=float)
    return 0.0004 * g**2 - 0.0129 * g + 0.2689


def Dp_from_g(g):
    g = np.asarray(g, dtype=float)
    return 2929.35 * g**(-0.40)


def Tc_from_g(g):
    g = np.asarray(g, dtype=float)
    return 72429.06 * g**(-1.90)


def g_from_I0(I_target, g_min=0.0, g_max=255.0):
    a = 0.0004
    b = -0.0129
    c = 0.2689 - float(I_target)
    disc = b * b - 4.0 * a * c

    if disc < 0.0:
        return float("nan")

    sqrt_disc = np.sqrt(disc)
    roots = [
        (-b + sqrt_disc) / (2.0 * a),
        (-b - sqrt_disc) / (2.0 * a),
    ]
    valid_roots = [r for r in roots if g_min <= r <= g_max]

    if not valid_roots:
        return float("nan")

    return float(max(valid_roots))


def t_for_required_Cd(g, required_Cd_um):
    Dp = float(Dp_from_g(g))
    Tc = float(Tc_from_g(g))
    return float(Tc * np.exp(float(required_Cd_um) / Dp))


I_OI_DEFAULT = 5.605
OMEGA_G_DEFAULT = 0.477


class OverhangColumn:
    def __init__(
        self,
        layer_thickness_um=50.0,
        I_OI=I_OI_DEFAULT,
        Omega_g=OMEGA_G_DEFAULT,
        void_depth_um=600.0,
        z_points=4801,
    ):
        self.layer_thickness_um = float(layer_thickness_um)
        self.I_OI = float(I_OI)
        self.Omega_g = float(Omega_g)
        self.void_depth_um = float(void_depth_um)
        self.z_points = int(z_points)

        self.n_exposures = 0
        self.prev_z = None
        self.prev_omega_memory = None
        self.prev_memory_mask = None
        self.prev_gel_boundary_um = 0.0
        self.prev_overcure_boundary_um = 0.0

        self.nominal_boundary_um = 0.0
        self.gel_boundary_um = 0.0
        self.overcure_boundary_um = 0.0
        self.gel_void_depth_um = 0.0
        self.overcure_void_depth_um = 0.0
        self.gel_only_band_depth_um = 0.0

    def expose_one_layer(self, g, t):
        self.n_exposures += 1

        nominal_boundary = self.n_exposures * self.layer_thickness_um
        z_max = nominal_boundary + self.void_depth_um
        z = np.linspace(0.0, z_max, self.z_points)

        g = float(g)
        t = float(t)
        I0 = float(I0_from_g(g))
        Dp = float(Dp_from_g(g))
        Tc = float(Tc_from_g(g))
        Cd = Dp * np.log(t / Tc) if t > Tc else -np.inf

        if self.n_exposures == 1:
            shifted_prev_omega = np.zeros_like(z)
            shifted_prev_mask = np.zeros_like(z, dtype=bool)
            mechanism_boundary = 0.0
        else:
            shifted_prev_omega = np.interp(
                z - self.layer_thickness_um,
                self.prev_z,
                self.prev_omega_memory,
                left=0.0,
                right=0.0,
            )
            shifted_prev_mask = (
                np.interp(
                    z - self.layer_thickness_um,
                    self.prev_z,
                    self.prev_memory_mask.astype(float),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
            mechanism_boundary = self.prev_gel_boundary_um + self.layer_thickness_um

        omega_new = np.zeros_like(z)
        I_local = np.zeros_like(z)

        if mechanism_boundary <= 0.0:
            omega_new = (t / Tc) * np.exp(-z / Dp)
            I_local = I0 * np.exp(-z / Dp)
        else:
            zone1 = z <= mechanism_boundary + 1e-12
            zone2 = ~zone1

            omega_new[zone1] = (t / Tc) * np.exp(-z[zone1] / Dp)
            I_local[zone1] = I0 * np.exp(-z[zone1] / Dp)

            I_void_in = I0 * np.exp(-mechanism_boundary / Dp)
            g_prime = g_from_I0(I_void_in)

            if np.isfinite(g_prime):
                Dp_z = float(Dp_from_g(g_prime))
                Tc_z = float(Tc_from_g(g_prime))
                omega_new[zone2] = (t / Tc_z) * np.exp(
                    -(z[zone2] - mechanism_boundary) / Dp_z
                )
                I_local[zone2] = I_void_in * np.exp(
                    -(z[zone2] - mechanism_boundary) / Dp_z
                )

        new_gel_mask = (I_local >= self.I_OI) & (omega_new >= self.Omega_g)
        memory_mask = shifted_prev_mask | new_gel_mask

        omega_memory = shifted_prev_omega.copy()
        omega_memory[memory_mask] += omega_new[memory_mask]
        omega_memory[~memory_mask] = 0.0

        if np.any(memory_mask):
            gel_boundary = float(z[memory_mask].max())
            memory_mask = z <= gel_boundary + 1e-12
        else:
            gel_boundary = 0.0

        newly_included = memory_mask & (omega_memory <= 0.0)
        omega_memory[newly_included] = omega_new[newly_included]
        omega_memory[~memory_mask] = 0.0

        overcure_mask = memory_mask & (omega_memory >= 1.0)
        if np.any(overcure_mask):
            overcure_boundary = float(z[overcure_mask].max())
        else:
            overcure_boundary = 0.0

        self.prev_z = z
        self.prev_omega_memory = omega_memory
        self.prev_memory_mask = memory_mask
        self.prev_gel_boundary_um = gel_boundary
        self.prev_overcure_boundary_um = overcure_boundary

        self.nominal_boundary_um = nominal_boundary
        self.gel_boundary_um = gel_boundary
        self.overcure_boundary_um = overcure_boundary
        self.gel_void_depth_um = max(0.0, gel_boundary - nominal_boundary)
        self.overcure_void_depth_um = max(0.0, overcure_boundary - nominal_boundary)
        self.gel_only_band_depth_um = max(0.0, gel_boundary - overcure_boundary)

        return {
            "n_exposures": self.n_exposures,
            "g": g,
            "t_s": t,
            "Cd_um": float(Cd),
            "nominal_boundary_um": self.nominal_boundary_um,
            "gel_boundary_um": self.gel_boundary_um,
            "overcure_boundary_um": self.overcure_boundary_um,
            "gel_void_depth_um": self.gel_void_depth_um,
            "overcure_void_depth_um": self.overcure_void_depth_um,
            "gel_only_band_depth_um": self.gel_only_band_depth_um,
        }
