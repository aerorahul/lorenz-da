#!/usr/bin/env python
"""
Variational Data Assimilation module

This module provides the main entry point for running variational
(3DVar/4DVar) data assimilation experiments.
"""

import sys
import numpy as np
from lorenzda.core import (
    Lorenz, get_IC, compute_B,
    check_varDA, update_varDA, create_diag
)
from lorenzda.core.module_IO import write_diag, create_obs


def run_variational_da(config_module):
    """
    Run variational data assimilation (3DVar/4DVar)

    Parameters
    ----------
    config_module : module
        Configuration module containing DA, varDA, model, H, R, Bc parameters
    """
    # Ensure reproducibility
    np.random.seed(0)

    # Extract configuration
    model = config_module.model
    DA = config_module.DA
    varDA = config_module.varDA
    restart = config_module.restart
    diag_file = config_module.diag_file
    H = config_module.H
    R = config_module.R
    Bc = config_module.Bc

    # Validate variational DA options
    check_varDA(DA, varDA)

    # Get initial conditions
    xt, xb = get_IC(model, restart)

    # Time between assimilations
    DA.tanal = model.dt * np.linspace(
        DA.t0,
        np.rint(DA.ntimes/model.dt),
        int(np.rint(DA.ntimes/model.dt)+1)
    )

    # Create diagnostic file
    create_diag(diag_file, model.Ndof)

    print('Cycling variational DA on the attractor...')

    for outer_loop in range(DA.maxouter):

        print(f'Outer loop {outer_loop+1:2d} of {DA.maxouter:2d}')

        for k in range(DA.nassim):
            print(f'  ========== assimilation cycle = {k+1:5d} ==========')

            # Advance truth with full nonlinear model
            xs = model.advance(xt, DA.tanal, perfect=True)
            xt = xs[-1, :].copy()
            ver = xt.copy()

            # Generate observations from truth
            y = np.squeeze(create_obs(model, varDA, xt, H, R))

            # Compute background error covariance
            B = compute_B(varDA, Bc, outer=outer_loop)

            # Update using variational method
            xa, niters = update_varDA(xb, B, y, R, H, varDA, model)

            # Write diagnostics to disk
            write_diag(
                diag_file.filename, k+1, outer_loop, ver, xb, xa,
                y, np.diag(H), np.diag(R), niters=niters
            )

            # Prepare for next cycle
            xb = xa.copy()

    print('... variational DA complete ...')
    return xa


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run variational data assimilation')
    parser.add_argument('config', help='Configuration module name (without .py)')
    args = parser.parse_args()

    # Dynamically import config module
    config = __import__(args.config, fromlist=[args.config])

    run_variational_da(config)
    sys.exit(0)
