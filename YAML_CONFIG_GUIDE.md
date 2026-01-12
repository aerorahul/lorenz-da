# YAML Configuration Guide

## Overview

The Lorenz-DA framework now supports both Python and YAML configuration files. YAML files provide a human-readable, structured format for specifying data assimilation parameters.

## File Format

Configuration files should be named with the pattern `{model}_{description}.yaml` or `{model}_{description}.yml`.

Examples:
- `L63_param_ensDA.yaml` - Ensemble DA for Lorenz 63
- `L96_param_varDA.yaml` - Variational DA for Lorenz 96
- `L63_param_hybDA.yml` - Hybrid DA for Lorenz 63

## Configuration Structure

### Model Configuration

```yaml
model:
  name: L63                              # Model name (L63 or L96)
  ndof: 3                                # Number of degrees of freedom
  parameters: [10.0, 28.0, 2.6667]       # Model parameters (σ, ρ, β for L63)
  dt: 0.01                               # Model time-step
  random_seed: 0                         # Random seed for reproducibility
```

**L63 Parameters:** `[sigma, rho, beta]`
- sigma: Prandtl number (~10)
- rho: Rayleigh number (~28)
- beta: Geometric factor (~8/3)

**L96 Parameters:** `[F, F+dF]`
- F: Forcing parameter
- F+dF: Perturbed forcing for truth generation

### Data Assimilation Configuration

```yaml
da:
  nassim: 200                    # Number of assimilation cycles
  ntimes: 0.05                   # Assimilation interval (non-dimensional time)
  t0: 0.0                        # Initial time
  nobs: 20                       # Number of observations (optional, for L96)
  do_hybrid: true                # Enable hybrid mode (optional)
  hybrid_weight: 0.5             # Hybrid weight [0=static, 1=ensemble]
  hybrid_recenter: true          # Recenter ensemble about variational solution
```

### Error Covariances

```yaml
# Model error covariance (process noise)
q: [0.0, 0.0, 0.0]     # Diagonal values, or null for defaults

# Observation operator
h: [1.0, 1.0, 1.0]     # Diagonal values, or null for identity

# Observation error covariance
r: [2.0, 2.0, 2.0]     # Diagonal values, or null for defaults

# Alternative for L96 with partial observations:
h_obs_indices: [0, 1, 2, ..., 19]  # Indices of observed variables
r_mean: 1.0                         # Mean observation error
r_noise: true                       # Add random noise to R
```

### Ensemble Configuration

```yaml
ensemble:
  update: 2              # Update method (0=None, 1=EnKF, 2=EnSRF, 3=EAKF)
  nens: 100              # Number of ensemble members
  init_ens_infl_fac: 1.0 # Initial ensemble inflation factor
  inflation:
    inflate: 1           # Inflation method (0=None, 1=Multiplicative, 2=Additive, 3=CovRelax, 4=SpreadRestoration)
    infl_fac: 1.1        # Inflation factor value
  localization:
    localize: 1          # Localization method (0=None, 1=Gaspari-Cohn, 2=Boxcar, 3=Ramped)
    cov_cutoff: 0.0625   # Covariance cutoff distance
    cov_trunc: 3         # Truncation parameter
```

### Variational Configuration

```yaml
variational:
  update: 1              # Method (1=3DVar, 2=4DVar)
  precondition: 1        # Preconditioning (0=None, 1=sqrtB, 2=FullB)
  maxouter: 1            # Number of outer loops
  minimization:
    maxiter: 1000        # Maximum iterations
    tolerance: 0.0001    # Convergence tolerance
  inflation:
    inflate: true        # Use inflation
    infl_fac: 1.85       # Inflation factor
    infl_adp: true       # Adaptive inflation
  localization:
    localize: 1          # Localization method
    cov_cutoff: 0.0625   # Cutoff distance
    cov_trunc: 40        # Truncation
  fv4d:
    window: 0.75         # Assimilation window length (4DVar only)
    offset: 0.25         # Time offset
    nobstimes: 4         # Number of observation times
```

### Output Configuration

```yaml
output:
  filename: L63_ensDA_diag.nc4
  attributes:
    model: L63
    sigma: 10.0
    rho: 28.0
    beta: 2.6667
    ntimes: 0.05
    dt: 0.01
    update: 2
    # ... other metadata
```

### Restart Configuration

```yaml
restart:
  time: null             # null for default, -1...-N or 0 1...N
  filename: ""           # Restart file path
```

## Complete Examples

### Example 1: L63 Ensemble DA

```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.6667]
  dt: 0.01
  random_seed: 0

da:
  nassim: 200
  ntimes: 0.05
  t0: 0.0

q: [0.0, 0.0, 0.0]
h: [1.0, 1.0, 1.0]
r: [2.0, 2.0, 2.0]

ensemble:
  update: 2           # EnSRF
  nens: 100
  init_ens_infl_fac: 1.0
  inflation:
    inflate: 1
    infl_fac: 1.1
  localization:
    localize: 0

output:
  filename: L63_ensDA_diag.nc4
  attributes:
    model: L63

restart:
  time: null
  filename: ""
```

### Example 2: L96 Variational DA

```yaml
model:
  name: L96
  ndof: 40
  parameters: [8.0, 8.4]
  dt: 0.0001
  random_seed: 0

da:
  nassim: 1000
  ntimes: 0.1
  t0: 0.0

h: null              # Use identity
r: null              # Use defaults

variational:
  update: 2          # 4DVar
  precondition: 1
  maxouter: 1
  minimization:
    maxiter: 1000
    tolerance: 0.0001
  localization:
    localize: 1
    cov_cutoff: 0.0625
    cov_trunc: 40
  fv4d:
    window: 0.75
    offset: 0.25
    nobstimes: 4

output:
  filename: L96_varDA_diag.nc4

restart:
  time: null
  filename: ""
```

### Example 3: Hybrid DA

```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.6667]
  dt: 0.01
  random_seed: 0

da:
  nassim: 200
  ntimes: 0.25
  t0: 0.0
  do_hybrid: true
  hybrid_weight: 0.5      # Balance between static and ensemble
  hybrid_recenter: true

ensemble:
  update: 2
  nens: 100
  inflation:
    inflate: 1
    infl_fac: 1.1
  localization:
    localize: 0

variational:
  update: 1               # 3DVar
  precondition: 1
  minimization:
    maxiter: 1000
    tolerance: 0.0001

output:
  filename: L63_hybDA_diag.nc4

restart:
  time: null
  filename: ""
```

## Usage

### From Command Line

Load a YAML configuration with the same command syntax:

```bash
# Using YAML file
lorenzda ensemble L63_param_ensDA

# Automatically finds L63_param_ensDA.yaml or L63_param_ensDA.py
```

### From Python Code

```python
from lorenzda.config import load_config_from_yaml

# Load YAML configuration
model, DA, Q, H, R, ensDA, varDA, diag_file, restart = \
    load_config_from_yaml('lorenzda/config/L63_param_ensDA.yaml')

# Use configuration objects directly
from lorenzda.apps import run_ensemble_da
run_ensemble_da(model, DA, Q, H, R, ensDA, diag_file, restart)
```

## Configuration Priority

When loading a configuration, the loader searches in this order:

1. `{config_name}.yaml` (current directory)
2. `{config_name}.yml` (current directory)
3. `lorenzda/config/{config_name}.yaml`
4. `lorenzda/config/{config_name}.yml`
5. `{config_name}` (Python module import)
6. `lorenzda.config.{config_name}` (Python module)
7. `lorenzda/config/{config_name}.py` (Python file)

## Null Values

In YAML, `null` means no value. When a configuration field is `null`:

- `q: null` → Uses default zero process noise
- `h: null` → Uses identity observation operator
- `r: null` → Uses default observation error

## Migration from Python Configs

To convert an existing Python config to YAML:

1. Identify the configuration objects (model, DA, Q, H, R, etc.)
2. Extract parameter values
3. Create equivalent YAML structure
4. Test with: `lorenzda <command> <config_name>`

Example Python config:

```python
model = type('',(),{})
model.Name = 'L63'
model.Ndof = 3
model.Par = [10.0, 28.0, 8.0/3.0]
model.dt = 0.01
```

Equivalent YAML:

```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.6667]
  dt: 0.01
```

## Tips and Best Practices

1. **Use consistent naming:** Follow the `{model}_{method}.yaml` convention
2. **Document parameters:** Add inline comments explaining parameter meanings
3. **Version control:** Keep YAML files in version control, not dynamic configs
4. **Validation:** Test configs with `lorenzda <command> --help`
5. **Defaults:** Use `null` for parameters that have sensible defaults
6. **Reproducibility:** Always specify `random_seed` for reproducible experiments

## Troubleshooting

**Error: "Cannot find configuration"**
- Check file name and location
- Ensure file has `.yaml` or `.yml` extension
- Verify YAML syntax with: `python -m yaml <file>`

**Error: "YAML parsing error"**
- Check indentation (YAML is whitespace-sensitive)
- Verify quotes around strings
- Use online YAML validators

**Wrong values loaded**
- Verify model `ndof` matches expected dimensions
- Check array lengths in parameters
- Ensure numerical types (not strings)
