"""
Lorenz Data Assimilation Framework

A framework for testing different data assimilation algorithms
(ensemble, variational, and hybrid methods) on simple Lorenz models.

Main components:
    apps        - Main application entry points
    cli         - Command-line interface
    config      - Configuration and parameter files
    core        - Core modules (Lorenz models, DA algorithms, I/O)
    utils       - Utility functions for plotting and analysis
"""


from . import core
#from . import apps
from . import config
from . import cli

__all__ = ['apps', 'cli', 'config', 'core']
