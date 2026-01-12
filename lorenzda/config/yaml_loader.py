"""
YAML Configuration Loader for Lorenz-DA

This module provides functionality to load YAML configuration files and
convert them to Python objects compatible with the existing data assimilation code.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Simple object to hold configuration attributes."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'Config({attrs})'


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Empty or invalid YAML file: {config_path}")

    return config_dict


def dict_to_object(data: Dict[str, Any]) -> Any:
    """
    Recursively convert dictionaries to Config objects.

    Args:
        data: Dictionary to convert

    Returns:
        Config object with nested Config objects for dict values
    """
    if not isinstance(data, dict):
        return data

    return Config(**{k: dict_to_object(v) for k, v in data.items()})


def build_observation_matrix(ndof: int, h_config: Optional[Any]) -> np.ndarray:
    """
    Build observation matrix from configuration.

    Args:
        ndof: Degrees of freedom (size of system)
        h_config: Observation configuration (list, None, or observation indices)

    Returns:
        Observation operator matrix
    """
    if h_config is None:
        # Default: observe all variables
        return np.eye(ndof)

    if isinstance(h_config, (list, np.ndarray)):
        # Assume it's a diagonal or list of values
        return np.diag(h_config)

    # Otherwise, assume it's None and return identity
    return np.eye(ndof)


def build_error_covariance(ndof: int, r_config: Optional[Any],
                          r_mean: Optional[float] = None,
                          r_noise: bool = False) -> np.ndarray:
    """
    Build error covariance matrix from configuration.

    Args:
        ndof: Degrees of freedom
        r_config: Error covariance configuration (list, None, or dict)
        r_mean: Mean value for error covariance
        r_noise: Whether to add random noise

    Returns:
        Error covariance matrix
    """
    if r_config is None:
        # Default: unit covariance
        r = np.ones(ndof)
    elif isinstance(r_config, (list, np.ndarray)):
        r = np.array(r_config)
    else:
        r = np.ones(ndof)

    if r_mean is not None:
        r = np.ones(ndof) * r_mean
        if r_noise:
            r += np.random.rand(ndof)

    return np.diag(r)


def build_process_noise(ndof: int, q_config: Optional[Any]) -> np.ndarray:
    """
    Build process noise (model error) covariance matrix.

    Args:
        ndof: Degrees of freedom
        q_config: Process noise configuration (list, None, or diagonal)

    Returns:
        Process noise covariance matrix
    """
    if q_config is None or q_config == []:
        # Default: zero process noise
        return np.diag(np.zeros(ndof))

    if isinstance(q_config, (list, np.ndarray)):
        return np.diag(q_config)

    return np.diag(np.zeros(ndof))


def yaml_to_config_objects(config_dict: Dict[str, Any]) -> tuple:
    """
    Convert a loaded YAML configuration to objects compatible with the DA code.

    Args:
        config_dict: Dictionary loaded from YAML

    Returns:
        Tuple of (model, DA, Q, H, R, and other config objects)
    """
    # Set random seed
    if 'model' in config_dict and 'random_seed' in config_dict['model']:
        np.random.seed(config_dict['model']['random_seed'])

    # Model configuration
    model_config = config_dict.get('model', {})
    ndof = model_config.get('ndof', 3)

    # Create model object
    model = Config(
        Name=model_config.get('name', 'L63'),
        Ndof=ndof,
        Par=model_config.get('parameters', [10.0, 28.0, 8.0/3.0]),
        dt=model_config.get('dt', 0.01)
    )

    # DA configuration
    da_config = config_dict.get('da', {})
    DA = Config(
        nassim=da_config.get('nassim', 200),
        ntimes=da_config.get('ntimes', 0.05),
        t0=da_config.get('t0', 0.0)
    )

    # Add hybrid parameters if present
    if 'do_hybrid' in da_config:
        DA.do_hybrid = da_config['do_hybrid']
        DA.hybrid_wght = da_config.get('hybrid_weight', 0.5)
        DA.hybrid_rcnt = da_config.get('hybrid_recenter', True)

    # Build matrices
    Q = build_process_noise(ndof, config_dict.get('q'))

    # Special handling for L96 with partial observations
    if 'h_obs_indices' in config_dict:
        H_indices = config_dict['h_obs_indices']
        H_diag = np.ones(ndof)
        H_diag[H_indices] = 1.0
        # Set unobserved variables to NaN
        for i in range(ndof):
            if i not in H_indices:
                H_diag[i] = np.nan
        H = np.diag(H_diag)
    else:
        H = build_observation_matrix(ndof, config_dict.get('h'))

    # Build R with special handling for L96
    if 'r_mean' in config_dict:
        R = build_error_covariance(ndof, None,
                                  r_mean=config_dict.get('r_mean'),
                                  r_noise=config_dict.get('r_noise', False))
    else:
        R = build_error_covariance(ndof, config_dict.get('r'))

    # Build ensemble configuration if present
    ensemble_config = config_dict.get('ensemble')
    if ensemble_config:
        ensDA = Config(
            update=ensemble_config.get('update', 2),
            Nens=ensemble_config.get('nens', 100),
            init_ens_infl_fac=ensemble_config.get('init_ens_infl_fac', 1.0)
        )

        # Inflation configuration
        infl_config = ensemble_config.get('inflation', {})
        ensDA.inflation = Config(
            inflate=infl_config.get('inflate', 0),
            infl_fac=infl_config.get('infl_fac', 1.01)
        )

        # Localization configuration
        loc_config = ensemble_config.get('localization', {})
        ensDA.localization = Config(
            localize=loc_config.get('localize', 0),
            cov_cutoff=loc_config.get('cov_cutoff', 1.0),
            cov_trunc=loc_config.get('cov_trunc', ndof)
        )
    else:
        ensDA = None

    # Build variational configuration if present
    variational_config = config_dict.get('variational')
    if variational_config:
        varDA = Config(
            update=variational_config.get('update', 1),
            precondition=variational_config.get('precondition', 1),
            maxouter=variational_config.get('maxouter', 1)
        )

        # Minimization
        min_config = variational_config.get('minimization', {})
        varDA.minimization = Config(
            maxiter=min_config.get('maxiter', 1000),
            tol=min_config.get('tolerance', 1e-4)
        )

        # Inflation (for variational)
        infl_config = variational_config.get('inflation', {})
        varDA.inflation = Config(
            inflate=infl_config.get('inflate', False),
            infl_fac=infl_config.get('infl_fac', 1.0),
            infl_adp=infl_config.get('infl_adp', False)
        )

        # Localization
        loc_config = variational_config.get('localization', {})
        varDA.localization = Config(
            localize=loc_config.get('localize', 0),
            cov_cutoff=loc_config.get('cov_cutoff', 1.0),
            cov_trunc=loc_config.get('cov_trunc', ndof)
        )

        # 4DVar configuration
        if varDA.update == 2:
            fv4d_config = variational_config.get('fv4d', {})
            varDA.fdvar = Config(
                window=fv4d_config.get('window', 0.25),
                offset=fv4d_config.get('offset', 0.5),
                nobstimes=fv4d_config.get('nobstimes', 1)
            )
    else:
        varDA = None

    # Output/diagnostic file configuration
    output_config = config_dict.get('output', {})
    diag_file = Config(
        filename=output_config.get('filename', 'lorenzda_diag.nc4'),
        attributes=output_config.get('attributes', {})
    )

    # Restart configuration
    restart_config = config_dict.get('restart', {})
    restart = Config(
        time=restart_config.get('time'),
        filename=restart_config.get('filename', '')
    )

    return model, DA, Q, H, R, ensDA, varDA, diag_file, restart


def load_config_from_yaml(config_path: str) -> tuple:
    """
    Load a complete configuration from a YAML file.

    This is the main entry point for loading configurations.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Tuple of configuration objects
    """
    config_dict = load_yaml_config(config_path)
    return yaml_to_config_objects(config_dict)
