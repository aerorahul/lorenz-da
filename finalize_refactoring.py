#!/usr/bin/env python3
"""
Setup script to finalize the Lorenz-DA refactoring

This script copies remaining files from the src/ directory to the new
refactored package structure.

Usage:
    python3 finalize_refactoring.py
"""

import shutil
from pathlib import Path
import sys


def main():
    """Copy files from src/ to refactored structure."""

    base_path = Path(__file__).parent
    src_path = base_path / "src"

    if not src_path.exists():
        print(f"Error: src/ directory not found at {src_path}")
        sys.exit(1)

    # Define file mappings
    mappings = {
        "core": [
            "module_Lorenz.py",
            "module_DA.py",
            "module_IO.py",
        ],
        "utils": [
            "plot_stats.py",
            "plot_Bs.py",
            "test_localization.py",
            "loop_LXX.py",
            "LXX_stats.py",
            "LXX_model.py",
        ],
        "config": [
            # Lorenz 63 parameter files
            "L63_param_truthDA.py",
            "L63_param_varDA.py",
            "L63_param_ensDA.py",
            "L63_param_hybDA.py",
            "L63_adjoint_sensitivity.py",
            "L63_param_ensDA.py",
            # Lorenz 96 parameter files
            "L96_param_truthDA.py",
            "L96_param_varDA.py",
            "L96_param_ensDA.py",
            "L96_param_hybDA.py",
            "L96_param_ensvarDA.py",
            "L96_param_hybensvarDA.py",
        ],
    }

    total_copied = 0
    total_skipped = 0

    for dest_dir, files in mappings.items():
        dest_path = base_path / "lorenzda" / dest_dir

        if not dest_path.exists():
            print(f"Creating {dest_path}...")
            dest_path.mkdir(parents=True, exist_ok=True)

        for file_name in files:
            src_file = src_path / file_name
            dest_file = dest_path / file_name

            if src_file.exists():
                if not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                    print(f"  ✓ Copied {file_name} to lorenzda/{dest_dir}/")
                    total_copied += 1
                else:
                    print(f"  - Skipped {file_name} (already exists)")
                    total_skipped += 1
            else:
                print(f"  ⚠ Not found: {file_name}")

    print(f"\nCopied {total_copied} files, skipped {total_skipped}")

    # Print summary
    print("\n" + "="*60)
    print("Refactoring complete!")
    print("="*60)

    print("\nDirectory structure:")
    print("""
lorenzda/
├── __init__.py
├── core/                    # Core modules
│   ├── __init__.py
│   ├── module_Lorenz.py
│   ├── module_DA.py
│   └── module_IO.py
├── apps/                    # Application entry points
│   ├── __init__.py
│   ├── truth_da.py
│   ├── ensemble_da.py
│   └── variational_da.py
├── cli/                     # Command-line interface
│   ├── __init__.py
│   └── main.py
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── L63_param_*.py
│   └── L96_param_*.py
└── utils/                   # Utility functions
    ├── __init__.py
    └── *.py (plotting, analysis)
    """)

    print("\nNext steps:")
    print("  1. Install the package: pip install -e .")
    print("  2. Run experiments: lorenzda truth L63_param_truthDA")
    print("  3. Or use programmatically: from lorenzda.apps import run_ensemble_da")

    return 0


if __name__ == "__main__":
    sys.exit(main())
