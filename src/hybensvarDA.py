#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# hybensvarDA.py - driver script for Hybrid Ensemble-Var DA
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@noaa.gov"
__copyright__ = "Copyright 2014, NOAA / NCEP / EMC"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import sys
import numpy             as     np
from   scipy             import linalg
from   module_Lorenz     import *
from   module_DA         import *
from   module_IO         import *
from   param_hybensvarDA import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble and variational data assimilation options
    check_hybensvarDA(DA,ensDA,varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xa = ( inflate_ensemble(Xa.T, ensDA.init_ens_infl_fac) ).T
    Xb = Xa.copy()
    xac = np.mean(Xa,axis=1)
    xbc = np.mean(Xb,axis=1)
    Xbwin = np.zeros((varDA.fdvar.nobstimes,model.Ndof,ensDA.Nens))
    Xawin = np.zeros((varDA.fdvar.nobstimes,model.Ndof,ensDA.Nens))

    # load climatological covariance once and for all ...
    Bc = read_clim_cov(model=model)

    # construct localization matrix once and for all ...
    L = localization_operator(model,ensDA.localization)
    A = np.kron(np.eye(ensDA.Nens),L)
    if ( varDA.precondition == 1 ):
        [U,S2,_] = np.linalg.svd(A, full_matrices=True, compute_uv=True)
        A = np.dot(U,np.diag(np.sqrt(S2)))

    nobs = model.Ndof*varDA.fdvar.nobstimes
    y    = np.tile(np.dot(H,xt),[varDA.fdvar.nobstimes,1])

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, nobs=nobs, nouter=DA.maxouter, hybrid=DA.do_hybrid)
    for outer in range(DA.maxouter):
        write_diag(diag_file.filename, 0, outer, xt, Xb.T, Xa.T, np.reshape(y,[nobs]), np.diag(H), np.diag(R), central_prior=xbc, central_posterior=xac, evratio=np.NaN, niters=np.NaN)

    for k in range(DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model; set verification values
        xs = model.advance(xt, varDA.fdvar.tbkgd, perfect=True)
        xt = xs[-1,:].copy()
        ver = xt.copy()

        # new observations from noise about truth
        y = create_obs(model,varDA,xt,H,R,yold=y)

        # advance the ensemble to the beginning of the window
        Xb = advance_ensemble(Xa, varDA.fdvar.tbkgd, model, perfect=False)
        Xbwin[0,:,:] = Xb.copy()
        Xa, evratio = update_ensDA(Xb, np.squeeze(y[0,:]), R, H, ensDA, model)
        Xawin[0,:,:] = Xa.copy()

        # advance the central analysis next with the full nonlinear model
        xs = model.advance(xac, varDA.fdvar.tbkgd, perfect=False)
        xbc = xs[-1,:].copy()

        # update the central background
        for outer in range(DA.maxouter):

            Xb = np.squeeze(Xbwin[0,:,:])
            Xa = np.squeeze(Xawin[0,:,:])

            # advance the ensemble through the window
            for i in range(1,varDA.fdvar.nobstimes):
                Xbwin[i,:,:] = advance_ensemble(Xbwin[i-1,:,:], varDA.fdvar.twind_obs, model, perfect=False)

            # compute static background error cov.
            Bs = compute_B(varDA,Bc,outer=outer)

            # construct block diagonal from static background error cov. and localization matrices
            B = linalg.block_diag( (1.0-DA.hybrid_wght) * Bs, DA.hybrid_wght * A )

            # construct D = [diag(columns of Xbp)]
            Xp = (Xb.T - np.mean(Xb,axis=1)).T/np.sqrt(ensDA.Nens-1)
            D = np.concatenate([np.diag(xi) for xi in Xp.T], axis=1)

            # construct C = [I,D]
            C = np.concatenate((np.eye(model.Ndof),D),axis=1)

            # update step
            xac, niters = update_hybensvarDA(xbc, B, C, np.squeeze(y), R, H, varDA, model)

            # write diagnostics to disk for each outer loop (at the beginning of the window)
            write_diag(diag_file.filename, k+1, outer, ver, Xb.T, Xa.T,
                    np.reshape(y,[nobs]), np.diag(H), np.diag(R), central_prior=xbc,
                    central_posterior=xac, evratio=evratio, niters=niters)

            Xbwin[0,:,:] = (Xb.T - np.mean(Xb,axis=1) + xac).T

        if ( DA.hybrid_rcnt ): Xa = (Xa.T - np.mean(Xa,axis=1) + xac).T

        # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
        if ( varDA.update == 2 ):
            xt  = model.advance(  xt,  varDA.fdvar.tanal,        perfect=True )[-1,:].copy()
            xac = model.advance(  xac, varDA.fdvar.tanal,        perfect=False)[-1,:].copy()
            Xa  = advance_ensemble(Xa, varDA.fdvar.tanal, model, perfect=False)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
