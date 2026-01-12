"""
Applications subpackage

Main entry points for running data assimilation experiments.
"""

from .truth_da import run_truth_da
from .ensemble_da import run_ensemble_da
from .variational_da import run_variational_da

__all__ = [
    'run_truth_da',
    'run_ensemble_da',
    'run_variational_da',
]
