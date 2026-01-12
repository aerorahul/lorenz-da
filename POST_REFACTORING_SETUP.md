# Post-Refactoring Setup Guide

## Quick Start

You now have a fully refactored Lorenz Data Assimilation package. Here's how to complete the setup and start using it.

## Step 1: Copy Remaining Files

Run the finalization script to copy files from `src/` to the new structure:

```bash
python finalize_refactoring.py
```

This will copy:
- Core modules (module_Lorenz.py, module_DA.py, module_IO.py) to `lorenzda/core/`
- Configuration files (L*_param_*.py) to `lorenzda/config/`
- Utility scripts (plot_*.py, etc.) to `lorenzda/utils/`

**Output:**
```
✓ Copied module_Lorenz.py to lorenzda/core/
✓ Copied module_DA.py to lorenzda/core/
✓ Copied module_IO.py to lorenzda/core/
✓ Copied L63_param_truthDA.py to lorenzda/config/
...
```

## Step 2: Install the Package

```bash
# Development install (recommended - changes are immediately reflected)
pip install -e .

# Standard install
pip install .

# With development tools
pip install -e ".[dev]"
```

## Step 3: Verify Installation

Test that everything works:

```bash
# Check CLI is available
lorenzda --help

# Try running a truth generation
lorenzda truth L63_param_truthDA

# Test module import
python -c "from lorenzda.core import Lorenz; print('Success!')"
```

## Step 4: Update Your Code (Optional)

If you have existing scripts that import from the old structure, update them:

```python
# Old imports
from module_Lorenz import Lorenz
from module_DA import check_ensDA

# New imports
from lorenzda.core import Lorenz, check_ensDA
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed examples.

## Project Structure Summary

```
lorenz-da/
├── lorenzda/                           # Main package
│   ├── __init__.py
│   ├── core/                           # Core modules
│   │   ├── __init__.py
│   │   ├── module_Lorenz.py          # Lorenz model
│   │   ├── module_DA.py              # DA algorithms
│   │   └── module_IO.py              # I/O utilities
│   ├── apps/                           # Applications
│   │   ├── __init__.py
│   │   ├── truth_da.py               # Truth generation
│   │   ├── ensemble_da.py            # Ensemble DA
│   │   └── variational_da.py         # Variational DA
│   ├── cli/                            # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py                   # CLI implementation
│   ├── config/                         # Configuration files
│   │   ├── __init__.py
│   │   ├── L63_param_truthDA.py
│   │   ├── L63_param_ensDA.py
│   │   ├── L63_param_varDA.py
│   │   ├── L96_param_truthDA.py
│   │   └── ... (more config files)
│   └── utils/                          # Utilities
│       ├── __init__.py
│       ├── plot_stats.py
│       ├── plot_Bs.py
│       ├── test_localization.py
│       └── ... (more utilities)
├── src/                                # Original code (keep for reference)
├── setup.py                            # Package setup
├── pyproject.toml                      # Modern Python packaging
├── REFACTORING.md                      # Refactoring overview
├── MIGRATION_GUIDE.md                  # Migration instructions
├── this_file                           # Post-refactoring setup
├── finalize_refactoring.py            # Setup script
└── README.md                           # Project documentation
```

## Usage Examples

### Via Command Line

```bash
# Generate truth
lorenzda truth L63_param_truthDA

# Run ensemble DA
lorenzda ensemble L63_param_ensDA

# Run variational DA
lorenzda variational L63_param_varDA

# See all options
lorenzda --help
```

### Via Python Script

```python
from lorenzda.apps import run_ensemble_da
from lorenzda.config import L63_param_ensDA

# Run ensemble DA programmatically
run_ensemble_da(L63_param_ensDA)
```

### Direct Module Usage

```python
from lorenzda.core import Lorenz, get_IC, update_ensDA

# Use modules directly (backward compatible)
model = Lorenz()
model.init(Name='L63')
xt, xa = get_IC(model, restart=None)
Xa, evratio = update_ensDA(Xb, y, R, H, ensDA, model)
```

## Package Features

### 1. **Modular Organization**
   - Core algorithms in `core/`
   - Applications in `apps/`
   - Configurations in `config/`
   - Utilities in `utils/`

### 2. **Professional CLI**
   - Single entry point: `lorenzda`
   - Subcommands for each method
   - Automatic config loading
   - Help and version information

### 3. **Flexible Execution**
   - Command-line for simple runs
   - Python API for complex workflows
   - Direct module import for research use

### 4. **Modern Python Packaging**
   - `setup.py` for compatibility
   - `pyproject.toml` for modern standards
   - Entry points for CLI commands
   - Support for installation via pip

### 5. **Documentation**
   - This setup guide
   - Migration guide for existing users
   - Refactoring documentation
   - API documentation in code

## Common Tasks

### Creating a Custom Experiment

```bash
# Copy a parameter file
cp lorenzda/config/L63_param_ensDA.py my_experiment.py

# Edit my_experiment.py with your parameters

# Run your experiment
lorenzda ensemble my_experiment
```

### Using Different Models

```bash
# Lorenz 63
lorenzda truth L63_param_truthDA
lorenzda ensemble L63_param_ensDA

# Lorenz 96
lorenzda truth L96_param_truthDA
lorenzda variational L96_param_varDA
```

### Writing Custom Code

```python
# Custom script using the package
from lorenzda.core import Lorenz, update_ensDA, create_diag
from lorenzda.config import L96_param_ensDA as config

# Your custom logic here
model = config.model
# ... your code ...
```

## Development Workflow

### Running Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
pytest --cov=lorenzda  # With coverage report
```

### Code Style

```bash
# Format code
black lorenzda/

# Check style
flake8 lorenzda/
```

### Building Distribution

```bash
# Create source and wheel distributions
pip install build
python -m build

# This creates dist/ directory with:
# - lorenzda-1.0.0.tar.gz (source)
# - lorenzda-1.0.0-py3-none-any.whl (wheel)
```

## Troubleshooting

### Issue: "command not found: lorenzda"

**Solution:** Package not installed
```bash
pip install -e .
```

### Issue: "ModuleNotFoundError: No module named 'lorenzda'"

**Solution:** Install the package
```bash
pip install -e .
```

### Issue: "Config file not found"

**Solutions:**
- Run from project root
- Config files must be in `lorenzda/config/`
- Or put them in current working directory

### Issue: Import errors in config files

**Solution:** Update imports to use `lorenzda.core`:
```python
# Update old imports
from lorenzda.core import Lorenz
from lorenzda.core import DataAssim
```

## Next Steps

1. ✅ Run `finalize_refactoring.py` to copy remaining files
2. ✅ Install the package: `pip install -e .`
3. ✅ Verify with: `lorenzda --help`
4. ✅ Try an example: `lorenzda truth L63_param_truthDA`
5. Update your own scripts and experiments
6. Archive or delete the old `src/` directory (optional, after verification)

## Documentation Files

- **README.md** - Project overview
- **REFACTORING.md** - Detailed refactoring information
- **MIGRATION_GUIDE.md** - How to update existing code
- **POST_REFACTORING_SETUP.md** - This file

## Key Improvements

✨ **Before (Original)**
- 33 files in flat `src/` directory
- Relative imports with sys.path manipulation
- Script-based execution
- Limited documentation structure

✨ **After (Refactored)**
- Organized into logical packages
- Clean, professional imports
- CLI tool + Python API
- Comprehensive documentation
- pip-installable package
- Better for distribution and collaboration

## Getting Help

1. **Installation Issues** → `pip install -e .` and check output
2. **Usage Questions** → `lorenzda --help` and see MIGRATION_GUIDE.md
3. **Code Understanding** → Check example configs in `lorenzda/config/`
4. **API Questions** → Look at code in `lorenzda/core/` and `lorenzda/apps/`

## Congratulations! 🎉

Your Lorenz-DA project is now professionally organized and ready to use:

```bash
# You can now do this:
lorenzda truth L63_param_truthDA
lorenzda ensemble L63_param_ensDA
lorenzda variational L96_param_varDA

# Share it easily:
pip install .

# Or install it in development:
pip install -e .
```

**That's it! Enjoy your refactored project!**
