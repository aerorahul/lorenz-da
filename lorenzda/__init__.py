"""
Lorenz Data Assimilation Framework

A comprehensive framework for testing different data assimilation algorithms
(ensemble, variational, and hybrid methods) on simple Lorenz models.

Main components:
    core        - Core modules (Lorenz models, DA algorithms, I/O)
    apps        - Main application entry points
    cli         - Command-line interface
    config      - Configuration and parameter files
    utils       - Utility functions for plotting and analysis
"""

__version__ = '1.0.0'
__author__ = 'Rahul Mahajan'
__email__ = 'rahul.mahajan@nasa.gov'

from . import core
from . import apps
from . import cli

__all__ = ['core', 'apps', 'cli']
