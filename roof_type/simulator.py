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
    roots = [
        (-b + sq) / (2.0 * a),
        (-b - sq) / (2.0 * a),
    ]
    valid_roots = [r for r in roots if g_min <= r <= g_max]
    if not valid_roots:
        return float("nan")

    return float(max(valid_roots))


def t_for_required_Cd(g, required_Cd_um):
    Dp = Dp_from_g(g)
    Tc = Tc_from_g(g)
    return float(Tc * np.exp(required_Cd_um / Dp))


I_OI_DEFAULT = 5.605
OMEGA_G_DEFAULT = 0.477


def simulate_gel_memory(
    g_list,
    t_list,
    I_OI=I_OI_DEFAULT,
    Omega_g=OMEGA_G_DEFAULT,
    layer_thickness_um=50.0,
    void_depth_um=600.0,
    z_points=4801,
):
    n_layers = len(g_list)
    if len(t_list) != n_layers:
        raise ValueError("g_list and t_list must have the same length.")

    prev_z = None
    prev_omega_memory = None
    prev_memory_mask = None
    prev_gel_boundary = 0.0

    nominal_boundary = 0.0
    gel_boundary = 0.0
    overcure_boundary = 0.0

    per_layer = []

    for j_idx in range(n_layers):
        j = j_idx + 1
        g = float(g_list[j_idx])
        t = float(t_list[j_idx])

        nominal_boundary = j * layer_thickness_um
        z = np.linspace(0.0, nominal_boundary + void_depth_um, z_points)

        I0 = float(I0_from_g(g))
        Dp = float(Dp_from_g(g))
        Tc = float(Tc_from_g(g))
        Cd = Dp * np.log(t / Tc) if t > Tc else -np.inf

        if j == 1:
            shifted_prev_omega = np.zeros_like(z)
            shifted_prev_mask = np.zeros_like(z, dtype=bool)
            mechanism_boundary = 0.0
        else:
            shifted_prev_omega = np.interp(
                z - layer_thickness_um,
                prev_z,
                prev_omega_memory,
                left=0.0,
                right=0.0,
            )
            shifted_prev_mask = (
                np.interp(
                    z - layer_thickness_um,
                    prev_z,
                    prev_memory_mask.astype(float),
                    left=0.0,
                    right=0.0,
                )
                > 0.5
            )
            mechanism_boundary = prev_gel_boundary + layer_thickness_um

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
                Dp_void = float(Dp_from_g(g_prime))
                Tc_void = float(Tc_from_g(g_prime))
                omega_new[out_zone] = (t / Tc_void) * np.exp(
                    -(z[out_zone] - mechanism_boundary) / Dp_void
                )
                I_local[out_zone] = I_void_in * np.exp(
                    -(z[out_zone] - mechanism_boundary) / Dp_void
                )

        new_gel_mask = (I_local >= I_OI) & (omega_new >= Omega_g)
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

        per_layer.append(
            {
                "j": j,
                "g": g,
                "t_s": t,
                "Cd_um": Cd,
                "nominal_boundary_um": nominal_boundary,
                "gel_boundary_um": gel_boundary,
                "overcure_boundary_um": overcure_boundary,
            }
        )

        prev_z = z
        prev_omega_memory = omega_memory
        prev_memory_mask = memory_mask
        prev_gel_boundary = gel_boundary

    final_gel_void_depth = max(0.0, gel_boundary - nominal_boundary)
    final_overcure_void_depth = max(0.0, overcure_boundary - nominal_boundary)
    final_gel_only_band_depth = max(0.0, gel_boundary - overcure_boundary)

    return {
        "final_nominal_boundary_um": nominal_boundary,
        "final_gel_boundary_um": gel_boundary,
        "final_overcure_boundary_um": overcure_boundary,
        "final_gel_void_depth_um": final_gel_void_depth,
        "final_overcure_void_depth_um": final_overcure_void_depth,
        "final_gel_only_band_depth_um": final_gel_only_band_depth,
        "per_layer": per_layer,
    }
