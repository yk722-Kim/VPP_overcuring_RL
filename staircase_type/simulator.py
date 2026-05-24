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
    a, b = 0.0004, -0.0129
    c = 0.2689 - float(I_target)
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("nan")

    sq = np.sqrt(disc)
    roots = [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]
    valid = [r for r in roots if g_min <= r <= g_max]
    if not valid:
        return float("nan")
    return max(valid)


def t_for_required_Cd(g, required_Cd_um):
    Dp = Dp_from_g(g)
    Tc = Tc_from_g(g)
    return float(Tc * np.exp(required_Cd_um / Dp))


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
        self.z = None
        self.omega_memory = None
        self.memory_mask = None

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

        if self.n_exposures == 1:
            shifted_omega = np.zeros_like(z)
            shifted_memory = np.zeros_like(z, dtype=bool)
            mechanism_boundary = 0.0
        else:
            shifted_omega = np.interp(
                z - self.layer_thickness_um,
                self.z,
                self.omega_memory,
                left=0.0,
                right=0.0,
            )
            shifted_memory = (
                np.interp(
                    z - self.layer_thickness_um,
                    self.z,
                    self.memory_mask.astype(float),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
            mechanism_boundary = self.gel_boundary_um + self.layer_thickness_um

        g = float(g)
        t = float(t)
        I0 = float(I0_from_g(g))
        Dp = float(Dp_from_g(g))
        Tc = float(Tc_from_g(g))

        omega_new = np.zeros_like(z)
        I_local = np.zeros_like(z)

        if mechanism_boundary <= 0.0:
            omega_new = (t / Tc) * np.exp(-z / Dp)
            I_local = I0 * np.exp(-z / Dp)
        else:
            in_zone = z <= mechanism_boundary + 1e-12
            out_zone = ~in_zone

            omega_new[in_zone] = (t / Tc) * np.exp(-z[in_zone] / Dp)
            I_local[in_zone] = I0 * np.exp(-z[in_zone] / Dp)

            I_void_in = I0 * np.exp(-mechanism_boundary / Dp)
            g_prime = g_from_I0(I_void_in)

            if np.isfinite(g_prime):
                Dp_z = float(Dp_from_g(g_prime))
                Tc_z = float(Tc_from_g(g_prime))
                dz = z[out_zone] - mechanism_boundary
                omega_new[out_zone] = (t / Tc_z) * np.exp(-dz / Dp_z)
                I_local[out_zone] = I_void_in * np.exp(-dz / Dp_z)

        new_gel_mask = (I_local >= self.I_OI) & (omega_new >= self.Omega_g)
        memory_mask = shifted_memory | new_gel_mask

        omega_memory = shifted_omega.copy()
        omega_memory[memory_mask] += omega_new[memory_mask]
        omega_memory[~memory_mask] = 0.0

        if np.any(memory_mask):
            gel_boundary = float(z[memory_mask].max())
            memory_mask = z <= gel_boundary + 1e-12
        else:
            gel_boundary = 0.0

        zero_in_memory = memory_mask & (omega_memory <= 0.0)
        omega_memory[zero_in_memory] = omega_new[zero_in_memory]
        omega_memory[~memory_mask] = 0.0

        overcure_mask = memory_mask & (omega_memory >= 1.0)
        if np.any(overcure_mask):
            overcure_boundary = float(z[overcure_mask].max())
        else:
            overcure_boundary = 0.0

        self.z = z
        self.omega_memory = omega_memory
        self.memory_mask = memory_mask

        self.nominal_boundary_um = nominal_boundary
        self.gel_boundary_um = gel_boundary
        self.overcure_boundary_um = overcure_boundary
        self.gel_void_depth_um = max(0.0, gel_boundary - nominal_boundary)
        self.overcure_void_depth_um = max(0.0, overcure_boundary - nominal_boundary)
        self.gel_only_band_depth_um = max(0.0, gel_boundary - overcure_boundary)


def simulate_gel_memory(
    g_list,
    t_list,
    I_OI=I_OI_DEFAULT,
    Omega_g=OMEGA_G_DEFAULT,
    layer_thickness_um=50.0,
):
    column = OverhangColumn(
        layer_thickness_um=layer_thickness_um,
        I_OI=I_OI,
        Omega_g=Omega_g,
    )

    per_layer = []
    for j, (g, t) in enumerate(zip(g_list, t_list), start=1):
        column.expose_one_layer(g, t)
        per_layer.append(
            {
                "j": j,
                "g": float(g),
                "t_s": float(t),
                "nominal_boundary_um": column.nominal_boundary_um,
                "gel_boundary_um": column.gel_boundary_um,
                "overcure_boundary_um": column.overcure_boundary_um,
                "gel_void_depth_um": column.gel_void_depth_um,
                "overcure_void_depth_um": column.overcure_void_depth_um,
                "gel_only_band_depth_um": column.gel_only_band_depth_um,
            }
        )

    return {
        "final_nominal_boundary_um": column.nominal_boundary_um,
        "final_gel_boundary_um": column.gel_boundary_um,
        "final_overcure_boundary_um": column.overcure_boundary_um,
        "final_gel_void_depth_um": column.gel_void_depth_um,
        "final_overcure_void_depth_um": column.overcure_void_depth_um,
        "final_gel_only_band_depth_um": column.gel_only_band_depth_um,
        "per_layer": per_layer,
    }
