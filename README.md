# Reinforcement-Learning-Driven Grayscale Modulation for Overcuring Suppression in Vat Photopolymerization

This repository contains the reinforcement-learning environments and cure-model simulators used for grayscale modulation in vat photopolymerization (VPP). The code implements a layer-by-layer cure model with two-zone optical penetration, oxygen-inhibition-gated gel memory, and PPO-based grayscale optimization for representative overhang geometries.

The implementation follows the manuscript:

**Reinforcement-Learning-Driven Grayscale Modulation for Overcuring Suppression in Vat Photopolymerization**

## Overview

Overcuring in VPP occurs when unintended polymerization extends beyond the target geometry and blocks or distorts void regions. This repository models that process using:

- grayscale-dependent incident light intensity
- grayscale-dependent penetration depth
- grayscale-dependent critical exposure time
- two-zone optical penetration through previously cured material and void resin
- oxygen-inhibition thresholding
- gel-memory accumulation across layer exposures
- reinforcement learning for grayscale schedule selection

The reinforcement-learning agent selects grayscale values that reduce both the retained gel-memory region and the fully solidified overcure region while penalizing long exposure times.

In the code, the variable `solid` or `solid_void_depth_um` corresponds to the manuscript quantity `z_overcure`.

## Repository Structure

Recommended structure:

```text
.
├── roof_type/
│   ├── simulator.py
│   ├── roof_env.py
│   └── train.py
├── staircase_type/
│   ├── simulator.py
│   ├── staircase_env.py
│   └── train.py
├── slant_type/
│   ├── simulator.py
│   ├── slant_env.py
│   └── train.py
└── README.md
```

Each geometry has its own environment and training script. Because the files use geometry-specific imports, keep each geometry in a separate folder or ensure that the correct `simulator.py`, environment file, and `train.py` are placed together before running.

## Installation

Create a Python environment and install the required packages:

```bash
pip install numpy gymnasium stable-baselines3
```

Optional, for TensorBoard logging:

```bash
pip install tensorboard
```

## Cure-Model Parameters

The simulator uses empirical grayscale-dependent optical and curing parameters:

```python
I0(g) = 0.0004*g**2 - 0.0129*g + 0.2689
Dp(g) = 2929.35*g**(-0.40)
Tc(g) = 72429.06*g**(-1.90)
```

The default oxygen-inhibition and gel-memory parameters are:

```python
I_OI = 5.605
Omega_g = 0.477
```

The grayscale action range is:

```python
g in {150, 151, ..., 255}
```

## Reward Definition

The reward follows the manuscript objective:

```text

R = -(z_gel + z_overcure) - beta * sum(t_j)

```

where:

- `z_gel` is the gel-memory void depth beyond the nominal build boundary.

- `z_overcure` is the fully solidified overcure depth beyond the nominal build boundary.

- `t_j` is the exposure time for layer `j`.

- `beta` is the exposure-time penalty coefficient.

Code-variable correspondence:

```text

gel_void_depth_um           -> z_gel

overcure_void_depth_um      -> z_overcure

gel_only_band_depth_um      -> z_gel - z_overcure

```

`gel_only_band_depth_um` is reported for diagnostics, but it is not used directly in the reward.

## Geometry-Specific Environments

### Roof-Type Overhang

Folder:

```bash
roof_type/
```

Run training:

```bash
cd roof_type
python train.py
```

The roof-type problem is treated as a single-episode schedule-selection problem. One action contains the full six-layer grayscale schedule.

Layer cure-depth constraints:

```text
Layer 1:      50 um
Layers 2-6:  100 um
```

The output includes the optimized grayscale schedule, corresponding exposure times, final gel depth, final overcure depth, and total exposure time.

### Staircase-Type Overhang

Folder:

```bash
staircase_type/
```

Run training:

```bash
cd staircase_type
python train.py
```

The staircase-type problem is treated as a step-based multi-column problem. The agent selects one grayscale value per layer. Each layer creates a new overhang column, and the current exposure is applied to all active overhang columns.

Layer cure-depth constraint:

```text
All layers: 50 um
```

Final outputs are reported per overhang column.

### Slant-Type Overhang

Folder:

```bash
slant_type/
```

Run training:

```bash
cd slant_type
python train.py
```

The slant-type problem is treated as a step-based multi-column problem. Each overhang column receives its birth exposure and one lateral-overlap exposure, except for the final overhang, which receives only its birth exposure.

Layer cure-depth constraint:

```text
All layers: 50 um
```

Final outputs are reported per overhang column.

## Training Settings

Default PPO training length:

```text
400,000 timesteps
```

The default training scripts use Stable-Baselines3 PPO with an MLP policy.

Example command with custom settings:

```bash
python train.py --total_timesteps 400000 --seed 0 --n_envs 8
```

## Output Interpretation

Typical final output fields include:

```text
g_history
t_history
final_gel_void_depth_um
final_solid_void_depth_um
final_gel_only_void_depth_um
total_exposure_time_s
```

For staircase-type and slant-type geometries, per-overhang results may also be reported:

```text
per_overhang_z_gel_um
per_overhang_z_overcure_um
per_overhang_z_gel_only_um
total_z_gel_um
total_z_overcure_um
total_objective_um
```

The key quantity corresponding to experimentally measured overcured void depth is:

```text
z_overcure = solid_void_depth_um
```

## Notes

- The simulator is a one-dimensional cure model for each overhang column.
- Lateral scattering, oxygen diffusion dynamics, and post-exposure dark cure are not explicitly modeled.
- The model is calibrated for the resin and printer settings described in the manuscript.
- The environments are intended to reproduce the manuscript-level grayscale optimization workflow, not to serve as a general-purpose VPP process simulator without recalibration.

## Citation

If this code is useful for your work, please cite the associated manuscript:

```bibtex
@article{kim_rl_vpp_grayscale,
  title   = {Reinforcement-Learning-Driven Grayscale Modulation for Overcuring Suppression in Vat Photopolymerization},
  author  = {Kim, Yuseok and Kang, Jeong-Hun and Seo, Min-Kyung and Park, Suk-Hee},
  journal = {Additive Manufacturing},
  year    = {TBD}
}
```

## License

Please specify the license before public release. For academic code release, consider adding a `LICENSE` file such as MIT, BSD-3-Clause, or another license required by your institution or journal.
