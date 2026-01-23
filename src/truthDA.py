#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# truthDA.py - driver script for truth for DA
###############################################################

###############################################################
from param_truthDA import *
from module_IO import *
from module_Lorenz import *
import numpy as np
import sys
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################
###############################################################

###############################################################


def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # get IC's
    [xt, _] = get_IC(model, restart)

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,
                                      np.rint(DA.ntimes / model.dt),
                                      int(np.rint(DA.ntimes / model.dt) + 1))

    # create diagnostic file
    #create_truth(diag_file, model.Ndof)
    #write_truth(
    #    diag_file.filename,
    #    0,
    #    xt,
    #    np.dot(
    #        H,
    #        xt),
    #    np.diag(H),
    #    np.diag(R))
    create_diag(diag_file, model.Ndof, nouter=1)
    write_diag(
        diag_file.filename,
        0,
        0,
        xt,
        xt,
        xt,
        np.dot(H, xt),
        np.diag(H),
        np.diag(R))

    print('running ON the attractor ...')

    for k in range(0, DA.nassim):

        print('========== assimilation time = %5d ========== ' % (k + 1))

        # advance truth with the full nonlinear model
        #xs = advance_model(model, xt, DA.tanal, perfect=True)
        xs = model.advance(xt, DA.tanal, perfect=True)
        xt = xs[-1, :].copy()

        # new observations from noise about truth; set verification values
        y = np.dot(H, xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))

        # write diagnostics to disk
        write_diag(
            diag_file.filename,
            k + 1,
            0,
            xt,
            xt,
            xt,
            y,
            np.diag(H),
            np.diag(R))

    print('... all done ...')
    sys.exit(0)
###############################################################


###############################################################
if __name__ == "__main__":
    main()
###############################################################
