"""
Configuration subpackage

Parameter files for different data assimilation experiments.
Configuration files are organized by model (L63, L96) and DA method.

Supports both Python (.py) and YAML (.yaml) configuration files.
YAML files are loaded using the yaml_loader module.
"""

from .yaml_loader import (
    load_config_from_yaml,
    load_yaml_config,
    yaml_to_config_objects,
    Config,
)

__all__ = [
    'load_config_from_yaml',
    'load_yaml_config',
    'yaml_to_config_objects',
    'Config',
]
