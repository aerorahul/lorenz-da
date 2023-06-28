#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# ensDA.py - driver script for ensemble DA
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import sys
import numpy         as     np
from   module_Lorenz import *
from   module_DA     import *
from   module_IO     import *
from   param_ensDA   import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble data assimilation options
    check_ensDA(DA,ensDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xa = ( inflate_ensemble(Xa.T, ensDA.init_ens_infl_fac) ).T
    Xb = Xa.copy()

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, nouter=1)
    write_diag(diag_file.filename, 0, 0, xt, Xb.T, Xa.T, np.dot(H,xt), np.diag(H), np.diag(R), evratio = np.NaN)

    print('Cycling ON the attractor ...')

    for k in range(DA.nassim):

        print('========== assimilation time = %5d ========== ' % (k+1))

        # advance truth with the full nonlinear model; set verification values
        xs = model.advance(xt, DA.tanal, perfect=True)
        xt = xs[-1,:].copy()
        ver = xt.copy()

        # new observations from noise about truth
        y = np.squeeze(create_obs(model, ensDA, xt, H, R))

        # advance analysis ensemble with the full nonlinear model
        Xb = advance_ensemble(Xa, DA.tanal, model, perfect=False)

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA, model)

        # write diagnostics to disk
        write_diag(diag_file.filename, k+1, 0, ver, Xb.T, Xa.T, y, np.diag(H), np.diag(R), evratio = evratio)

    print('... all done ...')
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
