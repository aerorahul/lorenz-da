# YAML Configuration System - Quick Reference

## What's New

The Lorenz-DA framework now supports **YAML configuration files** alongside traditional Python configurations.

### Quick Start

```bash
# Use YAML config (same command as before)
lorenzda ensemble L63_param_ensDA

# System auto-detects .yaml, .yml, or .py files
```

## File Locations

All YAML config files are in: `lorenzda/config/`

| File | Purpose |
|------|---------|
| `L63_param_truthDA.yaml` | Generate truth for L63 |
| `L63_param_varDA.yaml` | 3DVar/4DVar for L63 |
| `L63_param_ensDA.yaml` | EnKF/EnSRF/EAKF for L63 |
| `L63_param_hybDA.yaml` | Hybrid DA for L63 |
| `L96_param_truthDA.yaml` | Generate truth for L96 |
| `L96_param_varDA.yaml` | 3DVar/4DVar for L96 |
| `L96_param_ensDA.yaml` | EnKF/EnSRF/EAKF for L96 |
| `L96_param_hybDA.yaml` | Hybrid DA for L96 |
| `L96_param_ensvarDA.yaml` | Ensemble-Variational hybrid |
| `L96_param_hybensvarDA.yaml` | Hybrid ensemble-variational |

## Key Changes

### 1. New Dependency

Add `pyyaml` to requirements:
```bash
pip install pyyaml
```

Or install with development setup:
```bash
pip install -e ".[dev]"
```

### 2. New Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `lorenzda/config/yaml_loader.py` | YAML loading module | 350+ |
| `YAML_CONFIG_GUIDE.md` | Complete guide | 600+ |
| `YAML_CONVERSION_SUMMARY.md` | Implementation summary | 400+ |
| `CONFIG_FORMAT_COMPARISON.md` | Python vs YAML comparison | 300+ |

### 3. Updated Files

| File | Changes |
|------|---------|
| `lorenzda/cli/main.py` | Enhanced `load_config()` function |
| `lorenzda/config/__init__.py` | Exports YAML utilities |
| `pyproject.toml` | Added `pyyaml>=5.1` dependency |

## YAML Config Example

```yaml
# L63 Ensemble DA
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
  update: 2
  nens: 100
  inflation:
    inflate: 1
    infl_fac: 1.1
  localization:
    localize: 0

output:
  filename: L63_ensDA_diag.nc4

restart:
  time: null
  filename: ""
```

## Usage Modes

### Mode 1: CLI (No Code Changes)

```bash
# These all work identically now
lorenzda ensemble L63_param_ensDA        # Auto-detects format
lorenzda ensemble L63_param_ensDA.yaml   # Explicit YAML
lorenzda ensemble L63_param_ensDA.py     # Explicit Python (backward compat)
```

### Mode 2: Python - YAML Direct

```python
from lorenzda.config import load_config_from_yaml

# Load complete configuration
config = load_config_from_yaml('lorenzda/config/L63_param_ensDA.yaml')
model, DA, Q, H, R, ensDA, varDA, diag_file, restart = config

# Use in your code
from lorenzda.apps import run_ensemble_da
# run_ensemble_da(model, DA, Q, H, R, ensDA, diag_file, restart)
```

### Mode 3: Python - Auto Detection

```python
from lorenzda.cli.main import load_config

# Automatically finds and loads YAML or Python config
config = load_config('L63_param_ensDA')

# Returns appropriate config object (YAML loads return tuple)
```

### Mode 4: Python - Backward Compatible

```python
# Existing Python code still works
from lorenzda.config import L63_param_ensDA
run_ensemble_da(L63_param_ensDA)
```

## Configuration Search Order

When you run `lorenzda ensemble L63_param_ensDA`:

1. ✅ `L63_param_ensDA.yaml` (current directory)
2. ✅ `L63_param_ensDA.yml` (current directory)
3. ✅ `lorenzda/config/L63_param_ensDA.yaml`
4. ✅ `lorenzda/config/L63_param_ensDA.yml`
5. ✅ `L63_param_ensDA` (Python module import)
6. ✅ `lorenzda.config.L63_param_ensDA` (Python subpackage)
7. ✅ `lorenzda/config/L63_param_ensDA.py` (Python file)

**First match wins!**

## Parameter Reference

### Common Parameters

```yaml
# Model setup
model.name           # 'L63' or 'L96'
model.ndof           # Number of variables (3 for L63, 40 for L96)
model.parameters     # Model constants [σ, ρ, β] for L63 or [F, F+dF] for L96
model.dt             # Time step (0.01 for L63, 0.0001 for L96)
model.random_seed    # RNG seed for reproducibility

# DA cycle
da.nassim            # Number of assimilation cycles
da.ntimes            # Assimilation interval
da.t0                # Initial time

# Covariances
q                    # Process noise (model error)
h                    # Observation operator (null = identity)
r                    # Observation error
```

### Ensemble Parameters

```yaml
ensemble.update          # 1=EnKF, 2=EnSRF, 3=EAKF
ensemble.nens            # Number of ensemble members
ensemble.inflation.inflate    # Inflation method
ensemble.inflation.infl_fac   # Inflation factor
ensemble.localization.localize    # Localization type
ensemble.localization.cov_cutoff  # Cutoff distance
ensemble.localization.cov_trunc   # Truncation
```

### Variational Parameters

```yaml
variational.update           # 1=3DVar, 2=4DVar
variational.precondition     # 0=None, 1=sqrtB, 2=FullB
variational.minimization.maxiter     # Max iterations
variational.minimization.tolerance   # Convergence tolerance
variational.inflation.inflate        # Use inflation
variational.inflation.infl_fac       # Inflation value
variational.localization.*           # Localization settings
variational.fv4d.window              # 4DVar window length
```

## Hybrid DA Parameters

```yaml
da.do_hybrid         # Enable hybrid mode
da.hybrid_weight     # 0=static, 1=ensemble
da.hybrid_recenter   # Recenter ensemble about variational
```

## Special L96 Features

```yaml
# Partial observations (only first 20 of 40 observed)
h_obs_indices: [0, 1, 2, ..., 19]

# Observation error with noise
r_mean: 1.0
r_noise: true
```

## Documentation Files

| Document | Contents | Length |
|----------|----------|--------|
| **[YAML_CONFIG_GUIDE.md](YAML_CONFIG_GUIDE.md)** | Complete parameter reference with examples | 600+ lines |
| **[YAML_CONVERSION_SUMMARY.md](YAML_CONVERSION_SUMMARY.md)** | Implementation details and benefits | 400+ lines |
| **[CONFIG_FORMAT_COMPARISON.md](CONFIG_FORMAT_COMPARISON.md)** | Python vs YAML side-by-side | 300+ lines |

## Common Tasks

### Create a Custom Config

1. Copy an existing YAML file:
```bash
cp lorenzda/config/L63_param_ensDA.yaml my_experiment.yaml
```

2. Edit parameters:
```yaml
ensemble:
  nens: 50          # Changed from 100
  update: 3         # Changed to EAKF
```

3. Run it:
```bash
lorenzda ensemble my_experiment.yaml
```

### Convert Python Config to YAML

Extract parameter values from Python config and create equivalent YAML:

**Python:**
```python
model.Name = 'L63'
model.Ndof = 3
model.Par = [10.0, 28.0, 2.6667]
```

**YAML:**
```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.6667]
```

### Validate YAML Syntax

```bash
python -c "import yaml; print('Valid!') if yaml.safe_load(open('config.yaml')) else print('Invalid')"
```

### Debug Config Loading

```python
from lorenzda.config import load_config_from_yaml
import pprint

config_dict = load_config_from_yaml('L63_param_ensDA.yaml')
# This returns a tuple, so to see structure:
model, DA, Q, H, R, ensDA, varDA, diag_file, restart = config_dict
print("Model:", vars(model))
print("DA:", vars(DA))
```

## Backward Compatibility

✅ **100% backward compatible**

- All existing Python configs still work
- No code changes required for users
- Python and YAML configs can coexist
- Same CLI commands work for both

## Benefits Summary

| Benefit | Why It Matters |
|---------|----------------|
| **Human Readable** | Easy to understand and edit without Python knowledge |
| **Validation Ready** | Can use JSON Schema for automatic validation |
| **Version Control** | Changes are obvious in diffs (data, not code) |
| **Non-Programmer Friendly** | No Python syntax errors possible |
| **Tool Integration** | Can be generated/modified by external tools |
| **Easy Merging** | Combine multiple config files easily |
| **Comments** | Inline documentation of parameters |
| **Scalability** | Better for managing many experiments |

## Getting Started

### Step 1: Install PyYAML

```bash
pip install pyyaml
```

Or reinstall the package:
```bash
pip install -e .
```

### Step 2: Try It

```bash
# Run with YAML config
lorenzda ensemble L63_param_ensDA

# Run with specific YAML file
lorenzda truth lorenzda/config/L96_param_truthDA.yaml

# Both work identically to Python configs
```

### Step 3: Learn More

- Read [YAML_CONFIG_GUIDE.md](YAML_CONFIG_GUIDE.md) for complete parameter reference
- See [CONFIG_FORMAT_COMPARISON.md](CONFIG_FORMAT_COMPARISON.md) for Python vs YAML examples
- Check [YAML_CONVERSION_SUMMARY.md](YAML_CONVERSION_SUMMARY.md) for implementation details

## FAQ

**Q: Do I need to change my code?**
A: No! Python configs still work. You can mix and match.

**Q: How do I use YAML configs?**
A: Same command: `lorenzda ensemble L63_param_ensDA` - system auto-detects format.

**Q: Can I keep my Python configs?**
A: Yes! They continue to work indefinitely.

**Q: Is YAML config validation available?**
A: Not built-in yet, but YAML validation tools exist.

**Q: What if I have both .yaml and .py?**
A: YAML files take priority (searched first).

**Q: Can I mix parameters from both formats?**
A: No, but you can use one format per experiment.

## Support and Feedback

For issues with YAML configurations:

1. Check [YAML_CONFIG_GUIDE.md](YAML_CONFIG_GUIDE.md) troubleshooting section
2. Validate YAML syntax with online YAML validators
3. Compare with provided example YAML files
4. Check file paths and naming conventions

---

**Version:** 1.0
**Last Updated:** January 2026
**Status:** ✅ Production Ready
