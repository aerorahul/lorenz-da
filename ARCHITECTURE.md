# Project Architecture & Flow Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LORENZ-DA Package                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               CLI (Command Line Interface)                │   │
│  │  $ lorenzda ensemble L63_param_ensDA                      │   │
│  └────────────────┬──────────────────────────────────────────┘   │
│                   │                                               │
│  ┌────────────────▼──────────────────────────────────────────┐   │
│  │              Apps (Application Layer)                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │   │
│  │  │  Truth DA    │  │ Ensemble DA  │  │ Variational  │    │   │
│  │  │              │  │              │  │ DA           │    │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │   │
│  └────────────────┬──────────────────────────────────────────┘   │
│                   │                                               │
│  ┌────────────────▼──────────────────────────────────────────┐   │
│  │              Core Modules (Science)                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │   │
│  │  │ module_      │  │ module_      │  │ module_      │    │   │
│  │  │ Lorenz       │  │ DA           │  │ IO           │    │   │
│  │  │              │  │              │  │              │    │   │
│  │  │ - Models     │  │ - EnKF       │  │ - File I/O   │    │   │
│  │  │ - TLM/Adj    │  │ - 3DVar      │  │ - Diagnostics│    │   │
│  │  │ - Integration│  │ - 4DVar      │  │ - NetCDF     │    │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │   │
│  └────────────────┬──────────────────────────────────────────┘   │
│                   │                                               │
│  ┌────────────────▼──────────────────────────────────────────┐   │
│  │              Config Files (Parameters)                     │   │
│  │  ┌──────────────┐  ┌──────────────┐                       │   │
│  │  │ L63_param    │  │ L96_param    │  (+ many more)        │   │
│  │  │              │  │              │                       │   │
│  │  │ - Models     │  │ - Models     │                       │   │
│  │  │ - DA params  │  │ - DA params  │                       │   │
│  │  │ - Obs error  │  │ - Obs error  │                       │   │
│  │  └──────────────┘  └──────────────┘                       │   │
│  └────────────────┬──────────────────────────────────────────┘   │
│                   │                                               │
│  ┌────────────────▼──────────────────────────────────────────┐   │
│  │              Utilities (Analysis)                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                       │   │
│  │  │ Plotting     │  │ Analysis     │  (+ more utilities)    │   │
│  │  │              │  │              │                       │   │
│  │  │ - Attractor  │  │ - Statistics │                       │   │
│  │  │ - Errors     │  │ - Comparison │                       │   │
│  │  └──────────────┘  └──────────────┘                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Execution Flow

### CLI Execution Flow

```
User Input
    │
    ▼
┌─────────────────────────────┐
│   lorenzda ensemble config  │
└──────────────┬──────────────┘
               │
               ▼
        ┌─────────────────┐
        │  Load Config    │
        │  (param file)   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Call App Func  │
        │ run_ensemble_da │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Initialize     │
        │ - Model         │
        │ - DA params     │
        │ - Initial cond  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Main Loop      │
        │ - Advance model │
        │ - Generate obs  │
        │ - Update state  │
        │ - Write output  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Output Results │
        │  (NetCDF file)  │
        └─────────────────┘
```

### Python API Execution Flow

```
Python Script
    │
    ▼
┌──────────────────────────────┐
│   from lorenzda.apps import  │
│       run_ensemble_da        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   from lorenzda.config       │
│   import L63_param_ensDA     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   run_ensemble_da(config)    │
│   returns: Xa (analysis)     │
└──────────────────────────────┘
```

## File Organization

```
lorenz-da/
│
├── lorenzda/                       ← Main Package
│   │
│   ├── __init__.py                 ← Package initialization
│   │   (exports core, apps, cli)
│   │
│   ├── core/                        ← Core Science Modules
│   │   ├── __init__.py              (re-exports classes/functions)
│   │   ├── module_Lorenz.py         (Lorenz models, TLM, Adjoint)
│   │   ├── module_DA.py             (DA algorithms, updates, checks)
│   │   └── module_IO.py             (File I/O, diagnostics)
│   │
│   ├── apps/                        ← Application Programs
│   │   ├── __init__.py              (exports run_* functions)
│   │   ├── truth_da.py              (run_truth_da)
│   │   ├── ensemble_da.py           (run_ensemble_da)
│   │   └── variational_da.py        (run_variational_da)
│   │
│   ├── cli/                         ← Command Line Interface
│   │   ├── __init__.py              (exports main, create_parser)
│   │   └── main.py                  (CLI implementation)
│   │
│   ├── config/                      ← Configuration Files
│   │   ├── __init__.py              (package init)
│   │   ├── L63_param_truthDA.py
│   │   ├── L63_param_varDA.py
│   │   ├── L63_param_ensDA.py
│   │   ├── L96_param_truthDA.py
│   │   └── ...
│   │
│   └── utils/                       ← Utility Functions
│       ├── __init__.py              (package init)
│       ├── plot_stats.py
│       ├── plot_Bs.py
│       └── ...
│
├── setup.py                        ← Package setup script
├── pyproject.toml                  ← Modern Python config
│
├── REFACTORING_SUMMARY.md          ← Summary of changes
├── REFACTORING.md                  ← Detailed info
├── MIGRATION_GUIDE.md              ← User migration guide
├── POST_REFACTORING_SETUP.md       ← Setup instructions
├── finalize_refactoring.py         ← File copy script
│
└── src/                            ← Original code (keep for reference)
    ├── module_*.py
    ├── *DA.py
    └── ...
```

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                  External Dependencies                       │
│  numpy, scipy, matplotlib, netCDF4                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐         ┌──────────────────┐
│   module_Lorenz  │◄────┐   │    module_IO     │
│                  │     │   │                  │
│ - L63, L96       │     │   │ - File I/O       │
│ - TLM, Adjoint   │     │   │ - Diagnostics    │
│ - Integration    │     │   │ - NetCDF ops     │
└────────┬─────────┘     │   └────────┬─────────┘
         │               │            │
         │       ┌───────┴────────┐   │
         │       │                │   │
         ▼       ▼                ▼   ▼
┌──────────────────────────────────────────────┐
│          module_DA                           │
│                                              │
│ - EnKF, EnSRF, EAKF (Ensemble methods)      │
│ - 3DVar, 4DVar (Variational methods)        │
│ - Inflation, Localization                   │
│ - Covariance operations                     │
└────────┬─────────────────────────────────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌────────────────────┐    ┌──────────────────┐
│   App Functions    │    │  CLI Interface   │
│                    │    │                  │
│ - run_truth_da     │    │ - CLI commands   │
│ - run_ensemble_da  │◄───┤ - Config loading │
│ - run_variational  │    │ - Help system    │
└────────────────────┘    └──────────────────┘
```

## Data Flow for Ensemble DA

```
Initial Conditions (IC)
    │
    ├─ Truth: xt ──────────────┐
    │                           │
    └─ Ensemble: Xa ────┬──────┼─────────────┐
                        │      │             │
                        ▼      ▼             ▼
                   ┌──────────────────────────────┐
                   │   Cycle k (main loop)        │
                   └──────────────────────────────┘
                        │         │         │
        ┌───────────────┴──┐   ┌──┴──┐    ┌┴──────┐
        │                  │   │     │    │       │
        ▼                  ▼   ▼     ▼    ▼       ▼
    ┌──────────┐   ┌───────────┐  ┌────────────────┐
    │ Advance  │   │ Generate  │  │    Create      │
    │ Truth    │   │ Obs (y)   │  │  Obs Error (R) │
    │ (xt)     │   │           │  │                │
    └─────┬────┘   └─────┬─────┘  └────────┬───────┘
          │              │                 │
          ▼              ▼                 ▼
    ┌──────────────────────────────────────────┐
    │   Advance Ensemble (Xb)                  │
    │   Using nonlinear model                  │
    └────────────┬─────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────┐
    │   Assimilate Observations                │
    │   update_ensDA(Xb, y, R, H, ensDA, ...) │
    │                                          │
    │   - Compute prior estimate (H*Xb)       │
    │   - Compute innovations (y - H*Xb)      │
    │   - Update ensemble members             │
    │   - Apply inflation/localization        │
    │                                          │
    └────────────┬─────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────┐
    │   Write Output (Xa)                      │
    │   to NetCDF diagnostic file              │
    └────────────┬─────────────────────────────┘
                 │
    ┌────────────┴─────────────┐
    │                          │
    ▼                          ▼
  More cycles?           Final Results
  Yes → continue          (saved to disk)
  No  → exit
```

## Installation & Deployment Flow

```
Source Code
    │
    ▼
┌──────────────────────────┐
│  finalize_refactoring.py │  ◄─── Copies files from src/
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  setup.py / pyproject.toml
│  (Package metadata)      │
└─────────────┬────────────┘
              │
              ├─────────────────────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  pip install -e. │          │  pip install .   │
    │  (Development)   │          │  (Standard)      │
    └────────┬─────────┘          └────────┬─────────┘
             │                            │
             ▼                            ▼
    ┌──────────────────┐          ┌──────────────────┐
    │  Editable Mode   │          │  Normal Install  │
    │  (Changes live)  │          │  (System-wide)   │
    └─────────────────┘          └─────────────────┘
             │                            │
             └────────────┬───────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  lorenzda CLI    │
                │  Available!      │
                └──────────────────┘
```

## Key Concepts

### Separation of Concerns
```
Logic Layers:
├── Presentation (CLI)
├── Application (Apps)
├── Business Logic (Core)
├── Configuration (Config)
└── Utilities (Utils)
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `module_Lorenz` | Lorenz models, integration, TLM/Adjoint |
| `module_DA` | Data assimilation algorithms, analysis |
| `module_IO` | File I/O, NetCDF operations, diagnostics |

### Application Responsibilities

| App | Purpose |
|-----|---------|
| `truth_da.py` | Generate truth trajectories |
| `ensemble_da.py` | Run ensemble Kalman filter methods |
| `variational_da.py` | Run 3DVar/4DVar methods |

## Import Paths

```
User Code
    │
    ├─ Direct module use:
    │  from lorenzda.core.module_Lorenz import Lorenz
    │
    ├─ Via __init__ re-exports:
    │  from lorenzda.core import Lorenz
    │
    └─ Via app functions:
       from lorenzda.apps import run_ensemble_da
       from lorenzda.config import L63_param_ensDA
```

---

This architecture provides:
- ✅ Clear separation of concerns
- ✅ Easy testing and maintenance
- ✅ Professional package structure
- ✅ Flexible usage patterns (CLI or API)
- ✅ Scalability for future methods
