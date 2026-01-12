#!/usr/bin/env python
"""
Truth generation module for Data Assimilation

This module provides the main entry point for generating truth trajectories
for use in data assimilation experiments.
"""

import sys
import numpy as np
from lorenzda.core import Lorenz, get_IC
from lorenzda.core import create_diag
from lorenzda.core.module_IO import write_truth, advance_model


def run_truth_da(config_module):
    """
    Run truth data assimilation and generate truth trajectories

    Parameters
    ----------
    config_module : module
        Configuration module containing DA, model, H, R parameters
    """
    # Ensure reproducibility
    np.random.seed(0)

    # Extract configuration
    model = config_module.model
    DA = config_module.DA
    restart = config_module.restart
    truth_file = config_module.truth_file
    H = config_module.H
    R = config_module.R

    # Get initial conditions
    xt, _ = get_IC(model, restart)

    # Time between assimilations
    DA.tanal = model.dt * np.linspace(
        DA.t0,
        np.rint(DA.ntimes/model.dt),
        int(np.rint(DA.ntimes/model.dt)+1)
    )

    # Create diagnostic file
    create_diag(truth_file, model.Ndof)
    write_truth(truth_file.filename, 0, xt, np.dot(H,xt), np.diag(H), np.diag(R))

    print('Running truth on the attractor...')

    for k in range(DA.nassim):
        print(f'========== assimilation cycle = {k+1:5d} ==========')

        # Advance truth with full nonlinear model
        xs = advance_model(model, xt, DA.tanal, perfect=True)
        xt = xs[-1, :].copy()

        # Generate observations from truth with noise
        y = np.dot(H, xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))

        # Write diagnostics to disk
        write_truth(truth_file.filename, k+1, xt, y, np.diag(H), np.diag(R))

    print('... truth generation complete ...')
    return xt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate truth trajectory for DA')
    parser.add_argument('config', help='Configuration module name (without .py)')
    args = parser.parse_args()

    # Dynamically import config module
    config = __import__(args.config, fromlist=[args.config])

    run_truth_da(config)
    sys.exit(0)
