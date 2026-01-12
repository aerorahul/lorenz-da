# Lorenz Data Assimilation Framework - Refactored

## Overview

This is a refactored version of the Lorenz Data Assimilation project. The original monolithic script-based code has been reorganized into a modular Python package with a professional CLI interface.

### Key Improvements

- **Modular Structure**: Code is organized into logical packages
- **Professional CLI**: Unified command-line interface for running experiments
- **Configuration Management**: Separated configuration from code
- **Entry Points**: Executable entry point for system-wide installation
- **Package Installation**: Can be installed via pip for easy distribution
- **Better Imports**: Clear and organized imports throughout

## Project Structure

```
lorenz-da/
├── lorenzda/                    # Main package
│   ├── __init__.py
│   ├── core/                    # Core modules
│   │   ├── __init__.py
│   │   ├── module_Lorenz.py    # Lorenz model implementations
│   │   ├── module_DA.py        # DA algorithms
│   │   └── module_IO.py        # I/O utilities
│   ├── apps/                    # Application entry points
│   │   ├── __init__.py
│   │   ├── truth_da.py         # Truth generation
│   │   ├── ensemble_da.py      # Ensemble DA
│   │   └── variational_da.py   # Variational DA
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py             # CLI implementation
│   ├── config/                  # Configuration files
│   │   ├── __init__.py
│   │   ├── L63_param_*.py      # L63 parameters
│   │   └── L96_param_*.py      # L96 parameters
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── plot_stats.py        # Plotting utilities
├── setup.py                     # Package setup
├── pyproject.toml              # Modern Python packaging
└── README.md                    # This file
```

## Installation

### Development Installation

```bash
# Navigate to project directory
cd lorenz-da

# Install in editable mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Standard Installation

```bash
# After release
pip install lorenzda
```

## Usage

### Command-Line Interface

The refactored project provides a unified CLI:

```bash
# Generate truth trajectory
lorenzda truth L63_param_truthDA

# Run ensemble data assimilation
lorenzda ensemble L63_param_ensDA

# Run variational data assimilation (3DVar/4DVar)
lorenzda variational L63_param_varDA

# Run hybrid methods (future)
lorenzda hybrid L63_param_hybDA
lorenzda ensemble-var L96_param_ensvarDA
lorenzda hybrid-ensemble L96_param_hybensvarDA
```

### Programmatic Usage

```python
from lorenzda.apps import run_ensemble_da
import lorenzda.config.L63_param_ensDA as config

# Run ensemble DA
run_ensemble_da(config)
```

### As Python Module (backward compatible)

```python
from lorenzda.core import Lorenz, get_IC, update_ensDA

# Direct use of core modules
model = Lorenz()
model.init(Name='L63')
```

## Configuration

Configuration files are located in `lorenzda/config/`. They follow the naming convention:

- `L63_param_truthDA.py` - Lorenz 63 truth generation parameters
- `L63_param_ensDA.py` - Lorenz 63 ensemble DA parameters
- `L96_param_varDA.py` - Lorenz 96 variational DA parameters
- etc.

### Creating Custom Configurations

```python
# Copy an existing config file and modify
cp lorenzda/config/L63_param_ensDA.py my_config.py

# Then run with it
lorenzda ensemble my_config
```

## Core Modules

### `lorenzda.core.module_Lorenz`
- `Lorenz` class for model integration
- Functions for model advancement and I/O
- Plotting utilities

### `lorenzda.core.module_DA`
- Data assimilation algorithm implementations
- Classes: `DataAssim`, `VarDataAssim`, `EnsDataAssim`
- Update functions: `update_ensDA`, `update_varDA`
- Validation functions: `check_ensDA`, `check_varDA`

### `lorenzda.core.module_IO`
- File I/O operations
- Diagnostic file creation and writing

## Applications

Located in `lorenzda.apps/`, each app provides a main function:

- `run_truth_da()` - Generate truth trajectories
- `run_ensemble_da()` - Run ensemble-based DA
- `run_variational_da()` - Run variational DA (3DVar/4DVar)

## Utilities

Located in `lorenzda.utils/`:

- Plotting functions for visualization
- Analysis utilities
- Comparison scripts

## Supported Methods

### Implemented
- Ensemble Kalman Filter (EnKF)
- Ensemble Square Root Filter (EnSRF)
- Ensemble Adjustment Kalman Filter (EAKF)
- 3D Variational (3DVar)
- 4D Variational (4DVar)

### In Development
- Hybrid (ensemble-variational) methods
- Ensemble-Variational
- Hybrid Ensemble-Variational

## References

The original implementation is based on research in data assimilation, including:
- Lorenz (1963) - Three-variable model
- Lorenz & Emanuel (1998) - Forty-variable model

## Requirements

- Python ≥ 3.7
- NumPy ≥ 1.16.0
- SciPy ≥ 1.2.0
- Matplotlib ≥ 3.0.0
- NetCDF4 ≥ 1.4.0

## Development

### Running Tests

```bash
pytest
pytest --cov=lorenzda  # With coverage
```

### Code Style

```bash
black lorenzda/
flake8 lorenzda/
```

## License

GNU General Public License v3.0 - See LICENSE file for details

## Migration from Original Code

For users familiar with the original `src/` directory structure:

| Original | New Location |
|----------|--------------|
| `src/module_Lorenz.py` | `lorenzda/core/module_Lorenz.py` |
| `src/module_DA.py` | `lorenzda/core/module_DA.py` |
| `src/module_IO.py` | `lorenzda/core/module_IO.py` |
| `src/truthDA.py` | `lorenzda/apps/truth_da.py` |
| `src/ensDA.py` | `lorenzda/apps/ensemble_da.py` |
| `src/varDA.py` | `lorenzda/apps/variational_da.py` |
| `src/L*_param_*.py` | `lorenzda/config/L*_param_*.py` |

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please contact: rahul.mahajan@nasa.gov

## Project Status

**Version**: 1.0.0
**Status**: Beta / Active Development
