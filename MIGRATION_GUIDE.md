# Lorenz-DA Refactoring Guide

## What Changed

The Lorenz Data Assimilation project has been refactored from a flat script-based structure into a professional Python package. This document guides you through the changes and how to migrate your code.

## Key Changes

### 1. Project Structure

**Before (src/):**
```
src/
├── module_*.py          # Core modules
├── *DA.py              # Driver scripts (truthDA, ensDA, varDA, hybDA)
├── L*_param_*.py       # Configuration files
├── plot_*.py           # Plotting scripts
└── ... (33 files total)
```

**After (lorenzda/):**
```
lorenzda/
├── core/               # Core modules
├── apps/               # Application entry points
├── cli/                # Command-line interface
├── config/             # Configuration files
└── utils/              # Utility functions
```

### 2. File Locations

| Original Path | New Path |
|---|---|
| `src/module_Lorenz.py` | `lorenzda/core/module_Lorenz.py` |
| `src/module_DA.py` | `lorenzda/core/module_DA.py` |
| `src/module_IO.py` | `lorenzda/core/module_IO.py` |
| `src/truthDA.py` | `lorenzda/apps/truth_da.py` |
| `src/ensDA.py` | `lorenzda/apps/ensemble_da.py` |
| `src/varDA.py` | `lorenzda/apps/variational_da.py` |
| `src/L*_param_*.py` | `lorenzda/config/L*_param_*.py` |
| `src/plot_*.py` | `lorenzda/utils/plot_*.py` |

### 3. Import Changes

**Before (old style):**
```python
from module_Lorenz import Lorenz, get_IC
from module_DA import check_ensDA, update_ensDA
from module_IO import create_diag, write_diag
from param_ensDA import model, DA, ensDA
```

**After (new style):**
```python
# Option 1: Direct imports from core modules (backward compatible)
from lorenzda.core import Lorenz, get_IC
from lorenzda.core import check_ensDA, update_ensDA
from lorenzda.core import create_diag, write_diag

# Option 2: Import from lorenzda submodules
from lorenzda.core.module_Lorenz import Lorenz, get_IC
from lorenzda.core.module_DA import check_ensDA
from lorenzda.core.module_IO import create_diag

# Option 3: Use applications directly
from lorenzda.apps import run_ensemble_da
```

## Migration Guide

### For Users Running Experiments

**Old Way:**
```bash
cd src/
python ensDA.py
```

**New Way:**
```bash
# After installation
lorenzda ensemble L63_param_ensDA

# Or from project root
python -m lorenzda.cli ensemble L63_param_ensDA
```

### For Users Writing Scripts

**Old Script:**
```python
#!/usr/bin/env python
import sys
sys.path.insert(0, 'src/')

from module_Lorenz import Lorenz, get_IC
from module_DA import check_ensDA, update_ensDA
from module_IO import create_diag, write_diag
from L63_param_ensDA import model, DA, ensDA, ...

def main():
    # Your code here
    pass

if __name__ == '__main__':
    main()
```

**New Script (Option 1 - Minimal Changes):**
```python
#!/usr/bin/env python
from lorenzda.core import Lorenz, get_IC
from lorenzda.core import check_ensDA, update_ensDA
from lorenzda.core import create_diag, write_diag
from lorenzda.config.L63_param_ensDA import model, DA, ensDA, ...

def main():
    # Your code here (unchanged)
    pass

if __name__ == '__main__':
    main()
```

**New Script (Option 2 - Using Apps):**
```python
#!/usr/bin/env python
from lorenzda.apps import run_ensemble_da
from lorenzda.config import L63_param_ensDA as config

run_ensemble_da(config)
```

### For Developers

If you're modifying the core modules:

**Old Way:**
```bash
cd src/
python -m pytest test_*.py
```

**New Way:**
```bash
# From project root
pip install -e ".[dev]"
pytest
```

## Installation

### Option 1: Editable Install (Recommended for Development)

```bash
cd lorenz-da/
pip install -e .
```

This installs the package in "editable mode" - changes to the code are immediately reflected.

### Option 2: Standard Install

```bash
pip install .
```

### Option 3: With Development Tools

```bash
pip install -e ".[dev]"
# Includes: pytest, black, flake8
```

## CLI Usage

After installation, use the `lorenzda` command:

```bash
# Help
lorenzda --help
lorenzda truth --help

# Run experiments
lorenzda truth L63_param_truthDA
lorenzda ensemble L63_param_ensDA
lorenzda variational L63_param_varDA
```

### CLI Features

- **Configuration Loading**: Automatically finds config files in current dir or `lorenzda/config/`
- **Subcommands**: Different commands for different DA methods
- **Help**: `--help` flag shows detailed usage information
- **Future**: Support for hybrid methods (coming soon)

## Configuration Files

Configuration files have been moved to `lorenzda/config/` but the format is unchanged.

**Creating a Custom Config:**

```bash
# Copy an existing config
cp lorenzda/config/L63_param_ensDA.py my_experiment.py

# Edit my_experiment.py with your parameters

# Run with your config
lorenzda ensemble my_experiment
```

## Backward Compatibility

Most of the code is backward compatible:

1. **Module imports still work:**
   ```python
   from lorenzda.core.module_Lorenz import Lorenz
   # Same as before, just new package path
   ```

2. **Core functions unchanged:** All core DA algorithms work identically

3. **Parameter files compatible:** Configuration files require minimal or no changes

## Breaking Changes

These things changed and require updates:

1. **Driver scripts:** `truthDA.py`, `ensDA.py`, etc. are now in `lorenzda/apps/` with refactored main() functions
2. **Entry point:** Instead of running scripts directly, use the CLI or import the app functions
3. **Paths:** Old relative imports won't work without adding lorenzda to path

## Troubleshooting

### Import Error: "No module named 'lorenzda'"

**Solution:** Install the package
```bash
pip install -e .
```

### File Not Found: "L63_param_ensDA"

**Solution:** Files are in `lorenzda/config/`. Either:
- Run from project root
- Add to Python path: `export PYTHONPATH=.:$PYTHONPATH`
- Place configs in current working directory

### Config Module has Different Imports

**Solution:** Update imports in your config file:
```python
# Old
from module_Lorenz import Lorenz

# New
from lorenzda.core import Lorenz
```

## What's Better Now

1. **Cleaner Installation:** One standard `pip install` instead of path hacks
2. **Professional Structure:** Following Python packaging best practices
3. **CLI Tool:** Unified interface for running experiments
4. **Documentation:** Better organized and discoverable
5. **Testing:** Easier to write and run tests
6. **Distribution:** Can share as a package, not just source

## What's the Same

1. **Core algorithms:** Identical to original implementations
2. **Module interfaces:** Same functions and classes (just new import paths)
3. **Results:** Identical output and behavior
4. **File formats:** Configuration and output files unchanged

## Full Migration Checklist

- [ ] Install package: `pip install -e .`
- [ ] Update imports in your scripts
- [ ] Update file paths (config files moved to `lorenzda/config/`)
- [ ] Test one experiment with CLI: `lorenzda truth L63_param_truthDA`
- [ ] Update any deployment/documentation
- [ ] Archive old `src/` directory (keep as reference)

## Support

For issues with the refactoring:
1. Check this guide
2. Review example configs in `lorenzda/config/`
3. Look at example apps in `lorenzda/apps/`
4. Check the main README in project root

## Questions?

- Review [REFACTORING.md](REFACTORING.md) for project overview
- Check [README.md](README.md) for general documentation
- Examine example config files in `lorenzda/config/`
- Look at app source in `lorenzda/apps/`
