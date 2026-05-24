"""
Gel-memory + oxygen-inhibition simulator for the roof-type overhang.

Implements the same mechanism as Conv_graph_oxygen.py but stripped of
plotting / CSV output so it runs fast inside an RL environment.

Mechanism per layer j:
  1. Two-zone optics from z=0 across the previous gel-memory boundary.
       Inside the boundary       : Dp(g),    Tc(g)
       Beyond the boundary (void): Dp_z, Tc_z based on the attenuated
                                   incident I_void_in = I0 * exp(-B/Dp)
  2. New gel mask    : I_local(z) >= I_OI  AND  omega_new(z) >= Omega_g
  3. Memory mask     : shifted_prev_memory  OR  new_gel_mask
                       (then forced contiguous from z=0)
  4. Cells inside memory accumulate dose; cells outside reset to 0
     (peeled away by build-plate motion).
  5. Solid mask      : memory mask  AND  omega_memory >= 1
  6. Gel boundary    = max z in memory mask
     Solid boundary  = max z in solid mask
"""

import numpy as np


# -------- Empirical fits (paper Eqs. 5-7) --------
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
    """Solve I0(g') = I_target. Returns the larger valid root."""
    a, b = 0.0004, -0.0129
    c = 0.2689 - float(I_target)
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("nan")
    sq = np.sqrt(disc)
    r1 = (-b + sq) / (2.0 * a)
    r2 = (-b - sq) / (2.0 * a)
    valid = [r for r in (r1, r2) if g_min <= r <= g_max]
    if not valid:
        return float("nan")
    return max(valid)


def t_for_required_Cd(g, required_Cd_um):
    """Return t such that Cd(g, t) = required_Cd_um exactly."""
    Dp = Dp_from_g(g)
    Tc = Tc_from_g(g)
    return float(Tc * np.exp(required_Cd_um / Dp))


# -------- Fitted oxygen-inhibition / gel-memory parameters --------
I_OI_DEFAULT = 5.605    # W/m^2
OMEGA_G_DEFAULT = 0.477  # gel-formation dose threshold


def simulate_gel_memory(
    g_list, t_list,
    I_OI=I_OI_DEFAULT,
    Omega_g=OMEGA_G_DEFAULT,
    layer_thickness_um=50.0,
    void_depth_um=600.0,   # large enough to contain conventional ~237 um overcure
    z_points=4801,
):
    """
    Run the layer-by-layer gel-memory simulation.

    Returns a dict with the fields needed for RL reward and diagnostics.
    """
    n_layers = len(g_list)
    assert len(t_list) == n_layers

    prev_z = None
    prev_omega_memory = None
    prev_memory_mask = None
    prev_gel_boundary = 0.0
    prev_solid_boundary = 0.0

    nominal_boundary = 0.0
    gel_boundary = 0.0
    solid_boundary = 0.0

    per_layer = []

    for j_idx in range(n_layers):
        j = j_idx + 1
        g = float(g_list[j_idx])
        t = float(t_list[j_idx])
        nominal_boundary = j * layer_thickness_um
        plot_max = nominal_boundary + void_depth_um
        z = np.linspace(0.0, plot_max, z_points)

        I0 = float(I0_from_g(g))
        Dp = float(Dp_from_g(g))
        Tc = float(Tc_from_g(g))
        Cd = Dp * np.log(t / Tc) if t > Tc else -np.inf

        # Carry forward from previous layer (shift by one layer thickness)
        if j == 1:
            shifted_prev_omega = np.zeros_like(z)
            shifted_prev_mask = np.zeros_like(z, dtype=bool)
            pre_gel_boundary = 0.0
        else:
            shifted_prev_omega = np.interp(
                z - layer_thickness_um, prev_z, prev_omega_memory,
                left=0.0, right=0.0,
            )
            shifted_prev_mask = np.interp(
                z - layer_thickness_um, prev_z,
                prev_memory_mask.astype(float),
                left=0.0, right=0.0,
            ) > 0.5
            pre_gel_boundary = prev_gel_boundary + layer_thickness_um

        # Mechanism boundary: previous gel boundary in current frame
        mechanism_boundary = pre_gel_boundary  # 0.0 when j == 1

        # Two-zone optics
        omega_new = np.zeros_like(z)
        I_local = np.zeros_like(z)

        if mechanism_boundary <= 0.0:
            omega_new = (t / Tc) * np.exp(-z / Dp)
            I_local = I0 * np.exp(-z / Dp)
        else:
            in_mask = z <= mechanism_boundary + 1e-12
            out_mask = ~in_mask

            omega_new[in_mask] = (t / Tc) * np.exp(-z[in_mask] / Dp)
            I_local[in_mask] = I0 * np.exp(-z[in_mask] / Dp)

            I_void_in = I0 * np.exp(-mechanism_boundary / Dp)
            g_prime = g_from_I0(I_void_in)
            if not np.isfinite(g_prime):
                omega_new[out_mask] = 0.0
                I_local[out_mask] = 0.0
            else:
                Dp_z = float(Dp_from_g(g_prime))
                Tc_z = float(Tc_from_g(g_prime))
                omega_new[out_mask] = (t / Tc_z) * np.exp(
                    -(z[out_mask] - mechanism_boundary) / Dp_z
                )
                I_local[out_mask] = I_void_in * np.exp(
                    -(z[out_mask] - mechanism_boundary) / Dp_z
                )

        # Gel-memory update
        new_gel_mask = (I_local >= I_OI) & (omega_new >= Omega_g)
        memory_mask = shifted_prev_mask | new_gel_mask

        omega_memory = shifted_prev_omega.copy()
        omega_memory[memory_mask] += omega_new[memory_mask]
        omega_memory[~memory_mask] = 0.0

        # Force contiguity from z=0
        if np.any(memory_mask):
            gel_boundary = float(z[memory_mask].max())
            memory_mask = z <= gel_boundary + 1e-12
        else:
            gel_boundary = 0.0

        # Re-seed cells newly included by contiguity expansion
        zero_in_mem = memory_mask & (omega_memory <= 0.0)
        omega_memory[zero_in_mem] = omega_new[zero_in_mem]
        omega_memory[~memory_mask] = 0.0

        solid_mask = memory_mask & (omega_memory >= 1.0)
        if np.any(solid_mask):
            solid_boundary = float(z[solid_mask].max())
        else:
            solid_boundary = 0.0

        per_layer.append({
            "j": j,
            "g": g,
            "t_s": t,
            "Cd_um": Cd,
            "nominal_boundary_um": nominal_boundary,
            "gel_boundary_um": gel_boundary,
            "solid_boundary_um": solid_boundary,
        })

        prev_z = z
        prev_omega_memory = omega_memory
        prev_memory_mask = memory_mask
        prev_gel_boundary = gel_boundary
        prev_solid_boundary = solid_boundary

    final_gel_void_depth = max(0.0, gel_boundary - nominal_boundary)
    final_solid_void_depth = max(0.0, solid_boundary - nominal_boundary)
    # gel-only band = portion of overcured void that is gel but not solid
    final_gel_only_void_depth = max(0.0, gel_boundary - solid_boundary)

    return {
        "final_nominal_boundary_um": nominal_boundary,
        "final_gel_boundary_um": gel_boundary,
        "final_solid_boundary_um": solid_boundary,
        "final_gel_void_depth_um": final_gel_void_depth,
        "final_solid_void_depth_um": final_solid_void_depth,
        "final_gel_only_void_depth_um": final_gel_only_void_depth,
        "per_layer": per_layer,
    }
