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
    check_DA(DA)
    check_varDA(varDA)

    # get IC's
    [xt, xa] = get_IC(model, restart, Nens=None)
    xb = xa.copy()

    # construct localization matrix once and for all ...
    L = localization_operator(model,varDA.localization)

    if ( varDA.update == 2 ):
        varDA = FourDVar_obsWindow(model,DA,varDA)
    else:
        # time between assimilations
        DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    DA.nobs = model.Ndof   if (varDA.update == 1) else model.Ndof*varDA.fdvar.nobstimes
    y       = np.dot(H,xt) if (varDA.update == 1) else np.tile(np.dot(H,xt),[varDA.fdvar.nobstimes,1])
    # create diagnostic file and write initial conditions to the diagnostic file
    create_diag(diag_file, model.Ndof, nobs=DA.nobs, nouter=DA.maxouter)
    for outer in range(DA.maxouter):
        write_diag(diag_file.filename, 0, outer, xt, xb, xa, np.reshape(y,[DA.nobs]), np.diag(H), np.diag(R), niters=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model; set verification values
        if ( varDA.update == 1 ):
            xs = advance_model(model, xt, DA.tanal,          perfect=True)
        else:
            xs = advance_model(model, xt, varDA.fdvar.tbkgd, perfect=True)
        xt = xs[-1,:].copy()
        ver = xt.copy()

        # new observations from noise about truth
        y = create_obs(model,varDA,xt,H,R)

        # advance analysis with the full nonlinear model
        if   ( varDA.update == 1 ):
            # step to the next assimilation time (DA.tanal)
            xs = advance_model(model, xa, DA.tanal,          perfect=False)
        elif ( varDA.update == 2 ):
            # step to the beginning of the assimilation window (varDA.fdvar.tbkgd)
            xs = advance_model(model, xa, varDA.fdvar.tbkgd, perfect=False)
        xb = xs[-1,:].copy()

        for outer in range(DA.maxouter):

            # load B and make ready for update
            B = compute_B(model,varDA,outer=outer)

            # update step
            xa, niters = update_varDA(xb, B, y, R, H, varDA, model)

            # write diagnostics to disk for each outer loop
            write_diag(diag_file.filename, k+1, outer, ver, xb, xa, np.reshape(y,[DA.nobs]), np.diag(H), np.diag(R), niters=niters)

            # update prior for next outer loop
            xb = xa.copy()

        # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
        if ( varDA.update == 2 ):
            xs = advance_model(model, xt, varDA.fdvar.tanal, perfect=True )
            xt = xs[-1,:].copy()
            xs = advance_model(model, xa, varDA.fdvar.tanal, perfect=False)
            xa = xs[-1,:].copy()

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
