# Configuration Format Comparison

## Side-by-Side Comparison: Python vs YAML

### Example 1: Simple Truth Generation (L63)

#### Python Config (Original)
```python
#!/usr/bin/env python

import numpy as np

np.random.seed(0)

model      = type('',(),{})
model.Name = 'L63'
model.Ndof = 3
model.Par  = [10.0, 28.0, 8.0/3.0]
model.dt   = 1.0e-2

DA        = type('',(),{})
DA.nassim = 200
DA.ntimes = 0.05
DA.t0     = 0.0

Q = np.ones(model.Ndof)
Q = np.diag(Q) * 0.0

H = np.ones(model.Ndof)
H = np.diag(H)

R = np.ones(model.Ndof)
R = 2 * R
R = np.diag(R)

truth_file            = type('',(),{})
truth_file.filename   = 'L63_truthDA_diag.nc4'
truth_file.attributes = {'model': 'L63', 'sigma': 10.0, ...}

restart          = type('',(),{})
restart.time     = None
restart.filename = ''
```

**Lines:** ~30 | **Readability:** Medium | **Editable by:** Python developers

#### YAML Config (New)
```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.666666667]
  dt: 0.01
  random_seed: 0

da:
  nassim: 200
  ntimes: 0.05
  t0: 0.0

q: [0.0, 0.0, 0.0]
h: [1.0, 1.0, 1.0]
r: [2.0, 2.0, 2.0]

output:
  filename: L63_truthDA_diag.nc4
  attributes:
    model: L63
    sigma: 10.0

restart:
  time: null
  filename: ""
```

**Lines:** ~20 | **Readability:** High | **Editable by:** Anyone

---

### Example 2: Ensemble DA Configuration (L63)

#### Python Config
```python
#!/usr/bin/env python
import numpy as np
np.random.seed(0)

model      = type('',(),{})
model.Name = 'L63'
model.Ndof = 3
model.Par  = [10.0, 28.0, 8.0/3.0]
model.dt   = 1.0e-2

DA        = type('',(),{})
DA.nassim = 200
DA.ntimes = 0.05
DA.t0     = 0.0

Q = np.diag(np.ones(model.Ndof)) * 0.0
H = np.diag(np.ones(model.Ndof))
R = np.diag(np.ones(model.Ndof) * 2.0)

ensDA              = type('',(),{})
ensDA.inflation    = type('',(),{})
ensDA.localization = type('',(),{})
ensDA.update                  = 2
ensDA.Nens                    = 100
ensDA.init_ens_infl_fac       = 1.0
ensDA.inflation.inflate       = 1
ensDA.inflation.infl_fac      = 1.1
ensDA.localization.localize   = 0
ensDA.localization.cov_cutoff = 1.0
ensDA.localization.cov_trunc  = model.Ndof

diag_file            = type('',(),{})
diag_file.filename   = 'L63_ensDA_diag.nc4'
diag_file.attributes = {'model': 'L63', ...}

restart          = type('',(),{})
restart.time     = None
restart.filename = ''
```

**Lines:** ~35 | **Complexity:** High | **Errors possible:** Yes

#### YAML Config
```yaml
model:
  name: L63
  ndof: 3
  parameters: [10.0, 28.0, 2.666666667]
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
  init_ens_infl_fac: 1.0
  inflation:
    inflate: 1
    infl_fac: 1.1
  localization:
    localize: 0
    cov_cutoff: 1.0
    cov_trunc: 3

output:
  filename: L63_ensDA_diag.nc4
  attributes:
    model: L63

restart:
  time: null
  filename: ""
```

**Lines:** ~30 | **Complexity:** Low | **Errors possible:** Few

---

### Example 3: Variational DA Configuration (L96)

#### Python Config
```python
#!/usr/bin/env python
import numpy as np
from module_Lorenz import Lorenz
from module_DA import DataAssim, VarDataAssim

np.random.seed(0)
model = Lorenz()
DA    = DataAssim()
varDA = VarDataAssim()

model.init(Name='L96', Ndof=40, Par=[8.0,8.4], dt=1.e-4)

nassim   = 1000
ntimes   = 2.0*0.05
maxouter = 1
DA.init(nassim=nassim, ntimes=ntimes, maxouter=maxouter)

Q = np.diag(np.ones(model.Ndof)) * 0.0
H = np.diag(np.ones(model.Ndof))
R = np.diag(np.ones(model.Ndof))

update       = 2
precondition = 1
maxiter      = 1000
tol          = 1e-4
inflate      = True
infl_fac     = 1.85
infl_adp     = True
localize     = 1
cov_cutoff   = 0.0625
cov_trunc    = model.Ndof

window       = 0.75
offset       = 0.25
nobstimes    = 4

varDA.init(model, DA,
           update=update, precondition=precondition,
           maxiter=maxiter, tol=tol,
           inflate=inflate, infl_fac=infl_fac, infl_adp=infl_adp,
           localize=localize, cov_cutoff=cov_cutoff, cov_trunc=cov_trunc,
           window=window, offset=offset, nobstimes=nobstimes)

filename   = 'L96_varDA_diag.nc4'
attributes = {'model': 'L96', 'F': 8.0, ...}

restart          = type('',(),{})
restart.time     = None
restart.filename = ''
```

**Lines:** ~45 | **Dependencies:** Core modules | **Learning curve:** High

#### YAML Config
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
  maxouter: 1

h: null
r: null

variational:
  update: 2
  precondition: 1
  minimization:
    maxiter: 1000
    tolerance: 0.0001
  inflation:
    inflate: true
    infl_fac: 1.85
    infl_adp: true
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
  attributes:
    model: L96
    F: 8.0

restart:
  time: null
  filename: ""
```

**Lines:** ~40 | **Dependencies:** None | **Learning curve:** Minimal

---

## Comparison Matrix

| Aspect | Python Config | YAML Config |
|--------|---------------|-------------|
| **File Size** | Medium | Slightly smaller |
| **Readability** | Medium (Python syntax) | High (declarative) |
| **Edit by hand** | Yes (Python knowledge required) | Yes (intuitive) |
| **Type safety** | Runtime errors possible | Validation possible |
| **Comments** | Python-style `#` | YAML-style `#` |
| **Dependencies** | None | PyYAML |
| **Learning curve** | High (Python language) | Low (just data structure) |
| **Tool generation** | Possible | Easier |
| **Configuration merge** | Difficult | Easy |
| **Schema validation** | No | Yes (JSON Schema) |
| **Import other modules** | Yes | No |
| **Dynamically computed values** | Yes | No |
| **Nested structures** | Objects with dot notation | Natural YAML nesting |
| **Arrays/lists** | numpy arrays | YAML lists |
| **Version control** | Code (harder to track changes) | Data (easier to track changes) |

---

## Usage Comparison

### Loading Configuration

#### Python Way
```python
# Import module directly
from lorenzda.config import L63_param_ensDA as config
run_ensemble_da(config)

# Or use loader
from lorenzda.cli.main import load_config
config = load_config('L63_param_ensDA')
run_ensemble_da(config)
```

#### YAML Way
```python
# Load YAML file
from lorenzda.config import load_config_from_yaml
config = load_config_from_yaml('L63_param_ensDA.yaml')
# Returns tuple: (model, DA, Q, H, R, ensDA, varDA, diag_file, restart)
model, DA, Q, H, R, ensDA, varDA, diag_file, restart = config

# Or use the same loader (auto-detects format)
from lorenzda.cli.main import load_config
config = load_config('L63_param_ensDA')  # Finds .yaml automatically
```

### Command Line

Both formats work identically:

```bash
# YAML config
lorenzda ensemble L63_param_ensDA.yaml

# Python config
lorenzda ensemble L63_param_ensDA

# Without extension (auto-detects)
lorenzda ensemble L63_param_ensDA
```

---

## When to Use Each Format

### Use Python Configs When:
- You have complex parameter derivations
- You need dynamic computation at load time
- You want to import shared parameter definitions
- You're comfortable with Python syntax
- You need maximum flexibility

### Use YAML Configs When:
- You want human-readable configurations
- You need to share configs with non-programmers
- You want to validate configuration schema
- You prefer declarative over imperative
- You want easier version control tracking
- You're creating many similar experiments
- You want tool-generated configurations

---

## Recommendation

**Recommended approach:** Use YAML for most user-facing configurations, Python for specialized cases requiring computation.

- **Production environments:** YAML (easier to audit and validate)
- **Research/development:** Either (Python if computation needed)
- **Deployment/DevOps:** YAML (integrates with configuration systems)
- **Shared experiments:** YAML (more portable and version-control-friendly)
