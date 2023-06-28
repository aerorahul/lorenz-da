#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# varDA.py - driver script for variational DA
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
from   param_varDA   import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid variational data assimilation options
    check_varDA(DA,varDA)

    # get IC's
    [xt, xa] = get_IC(model, restart, Nens=None)
    xb = xa.copy()

    # Load climatological covariance once and for all ...
    Bc = read_clim_cov(model=model,norm=True)

    nobs = model.Ndof*varDA.fdvar.nobstimes
    y    = np.tile(np.dot(H,xt),[varDA.fdvar.nobstimes,1])

    # create diagnostic file and write initial conditions to the diagnostic file
    create_diag(diag_file, model.Ndof, nobs=nobs, nouter=DA.maxouter)
    for outer in range(DA.maxouter):
        write_diag(diag_file.filename, 0, outer, xt, xb, xa, np.reshape(y,[nobs]), np.diag(H), np.diag(R), niters=np.NaN)

    print('Cycling ON the attractor ...')

    for k in range(DA.nassim):

        print('========== assimilation time = %5d ========== ' % (k+1))

        # advance truth with the full nonlinear model; set verification values
        xs = model.advance(xt, varDA.fdvar.tbkgd, perfect=True)
        xt = xs[-1,:].copy()
        ver = xt.copy()

        # new observations from noise about truth
        y = create_obs(model,varDA,xt,H,R,yold=y)

        # advance analysis with the full nonlinear model
        xs = model.advance(xa, varDA.fdvar.tbkgd, perfect=False)
        xb = xs[-1,:].copy()

        for outer in range(DA.maxouter):

            # compute static background error cov.
            Bs = compute_B(varDA,Bc,outer=outer)

            # update step
            xa, niters = update_varDA(xb, Bs, y, R, H, varDA, model)

            # write diagnostics to disk for each outer loop (at the beginning of the window)
            write_diag(diag_file.filename, k+1, outer, ver, xb, xa, np.reshape(y,[nobs]), np.diag(H), np.diag(R), niters=niters)

            # update prior for next outer loop
            xb = xa.copy()

        # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
        if ( varDA.update == 2 ):
            xs = model.advance(xt, varDA.fdvar.tanal, perfect=True )
            xt = xs[-1,:].copy()
            xs = model.advance(xa, varDA.fdvar.tanal, perfect=False)
            xa = xs[-1,:].copy()

    print('... all done ...')
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
