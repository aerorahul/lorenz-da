#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# hybDA.py - driver script for hybrid DA
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
from   param_hybDA   import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble and variational data assimilation options
    check_hybDA(DA,ensDA,varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xa = ( inflate_ensemble(Xa.T, ensDA.init_ens_infl_fac) ).T
    Xb = Xa.copy()
    xac = np.mean(Xa,axis=1)
    xbc = np.mean(Xb,axis=1)

    # load climatological covariance once and for all ...
    Bc = read_clim_cov(model)

    # construct localization matrix once and for all ...
    L = localization_operator(model,ensDA.localization)

    nobs = model.Ndof*varDA.fdvar.nobstimes
    y    = np.tile(np.dot(H,xt),[varDA.fdvar.nobstimes,1])

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, nobs=nobs, nouter=DA.maxouter, hybrid=DA.do_hybrid)
    for outer in range(DA.maxouter):
        write_diag(diag_file.filename, 0, outer, xt, Xb.T, Xa.T, np.reshape(y,[nobs]), np.diag(H), np.diag(R), central_prior=xbc, central_posterior=xac, evratio=np.NaN, niters=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model; set verification values
        xs = model.advance(xt, varDA.fdvar.tbkgd, perfect=True)
        xt = xs[-1,:].copy()
        ver = xt.copy()

        # new observations from noise about truth
        y = create_obs(model,varDA,xt,H,R)

        # advance the ensemble first with the full nonlinear model
        Xb = advance_ensemble(Xa, varDA.fdvar.tbkgd, model, perfect=False)

        # compute flow-dependent background error covariance from ensemble
        Be = np.cov(Xb, ddof=1)

        # update the ensemble
        Xa, evratio = update_ensDA(Xb, np.squeeze(y[0,:]), R, H, ensDA, model)

        # advance the central analysis next with the full nonlinear model
        xs = model.advance(xac, varDA.fdvar.tbkgd, perfect=False)
        xbc = xs[-1,:].copy()

        # update the central background
        for outer in range(DA.maxouter):

            # compute static background error cov.
            Bs = compute_B(varDA,Bc,outer=outer)

            # blend covariance from flow-dependent (ensemble) and static (climatology)
            B = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * (Be*L)
            if ( varDA.precondition == 1 ):
                [U,S2,_] = np.linalg.svd(B, full_matrices=True, compute_uv=True)
                B = np.dot(U,np.diag(np.sqrt(S2)))

            # update step
            xac, niters = update_varDA(xbc, B, y, R, H, varDA, model)

            # write diagnostics to disk for each outer loop (at the beginning of the window)
            write_diag(diag_file.filename, k+1, outer, ver, Xb.T, Xa.T, y, np.diag(H), np.diag(R), central_prior=xbc, central_posterior=xac, evratio=evratio, niters=niters)

        # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
        if ( varDA.update == 2 ):
            xs  = model.advance(xt,  varDA.fdvar.tanal, perfect=True )
            xt  = xs[-1,:].copy()
            xs  = model.advance(xac, varDA.fdvar.tanal, perfect=False)
            xac = xs[-1,:].copy()

            # update the ensemble through the window
            for i in range(varDA.fdvar.nobstimes):

                Xb = advance_ensemble(Xa, varDA.fdvar.twind_obs, model, perfect=False)
                Xa, evratio = update_ensDA(Xb, y[i,:], R, H, ensDA, model)

                if ( varDA.fdvar.tb + varDA.fdvar.twind_obsIndex[i] > varDA.fdvar.ta ): break

        # recenter ensemble about central analysis
        if ( DA.hybrid_rcnt ): Xa = (Xa.T - np.mean(Xa,axis=1) + xac).T

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
