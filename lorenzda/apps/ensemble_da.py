#!/usr/bin/env python
"""
Ensemble Data Assimilation module

This module provides the main entry point for running ensemble-based
data assimilation experiments.
"""

import sys
import numpy as np
from lorenzda.core import (
    Lorenz, get_IC, advance_ensemble, inflate_ensemble,
    check_ensDA, update_ensDA, create_diag
)
from lorenzda.core.module_IO import write_diag, create_obs, advance_model


def run_ensemble_da(config_module):
    """
    Run ensemble data assimilation

    Parameters
    ----------
    config_module : module
        Configuration module containing DA, ensDA, model, H, R parameters
    """
    # Ensure reproducibility
    np.random.seed(0)

    # Extract configuration
    model = config_module.model
    DA = config_module.DA
    ensDA = config_module.ensDA
    restart = config_module.restart
    diag_file = config_module.diag_file
    H = config_module.H
    R = config_module.R

    # Validate ensemble DA options
    check_ensDA(DA, ensDA)

    # Get initial conditions
    xt, Xa = get_IC(model, restart, Nens=ensDA.Nens)
    Xa = (inflate_ensemble(Xa.T, ensDA.init_ens_infl_fac)).T
    Xb = Xa.copy()

    # Time between assimilations
    DA.tanal = model.dt * np.linspace(
        DA.t0,
        np.rint(DA.ntimes/model.dt),
        int(np.rint(DA.ntimes/model.dt)+1)
    )

    # Create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, nouter=1)
    write_diag(
        diag_file.filename, 0, 0, xt, Xb.T, Xa.T,
        np.dot(H, xt), np.diag(H), np.diag(R), evratio=np.NaN
    )

    print('Cycling ensemble DA on the attractor...')

    for k in range(DA.nassim):
        print(f'========== assimilation cycle = {k+1:5d} ==========')

        # Advance truth with full nonlinear model
        xs = model.advance(xt, DA.tanal, perfect=True)
        xt = xs[-1, :].copy()
        ver = xt.copy()

        # Generate observations
        y = np.squeeze(create_obs(model, ensDA, xt, H, R))

        # Advance analysis ensemble with full nonlinear model
        Xb = advance_ensemble(Xa, DA.tanal, model, perfect=False)

        # Update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA, model)

        # Write diagnostics to disk
        write_diag(
            diag_file.filename, k+1, 0, ver, Xb.T, Xa.T,
            y, np.diag(H), np.diag(R), evratio=evratio
        )

    print('... ensemble DA complete ...')
    return Xa


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ensemble data assimilation')
    parser.add_argument('config', help='Configuration module name (without .py)')
    args = parser.parse_args()

    # Dynamically import config module
    config = __import__(args.config, fromlist=[args.config])

    run_ensemble_da(config)
    sys.exit(0)
