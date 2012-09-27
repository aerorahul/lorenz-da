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
    check_DA(DA)
    check_ensDA(ensDA)
    if ( DA.do_hybrid ): check_varDA(varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xb = Xa.copy()
    if ( DA.do_hybrid ):
        xac = np.mean(Xa,axis=1)
        xbc = np.mean(Xb,axis=1)
        Xbwin = Xa.copy()

    # load climatological covariance once and for all ...
    if ( DA.do_hybrid ): Bs = varDA.inflation.infl_fac * read_clim_cov(model)

    # construct localization matrix once and for all ...
    L = localization_operator(model,ensDA.localization)

    if ( DA.do_hybrid and varDA.update == 2 ):
        # check length of assimilation window
        if ( varDA.fdvar.offset * DA.ntimes + varDA.fdvar.window - DA.ntimes < 0.0 ):
            print '4DVar assimilation window is too short'
            sys.exit(2)

        # time index from analysis to ... background, next analysis, end of window, window
        varDA.fdvar.tb = np.int(np.rint(varDA.fdvar.offset * DA.ntimes/model.dt))
        varDA.fdvar.ta = np.int(np.rint(DA.ntimes/model.dt))
        varDA.fdvar.tf = np.int(np.rint((varDA.fdvar.offset * DA.ntimes + varDA.fdvar.window)/model.dt))
        varDA.fdvar.tw = varDA.fdvar.tf - varDA.fdvar.tb

        # time vector from analysis to ... background, next analysis, end of window, window
        varDA.fdvar.tbkgd = np.linspace(DA.t0,varDA.fdvar.tb,   varDA.fdvar.tb   +1) * model.dt
        varDA.fdvar.tanal = np.linspace(DA.t0,varDA.fdvar.ta-varDA.fdvar.tb,varDA.fdvar.ta-varDA.fdvar.tb+1) * model.dt
        varDA.fdvar.tfore = np.linspace(DA.t0,varDA.fdvar.tf,   varDA.fdvar.tf   +1) * model.dt
        varDA.fdvar.twind = np.linspace(DA.t0,varDA.fdvar.tw,   varDA.fdvar.tw   +1) * model.dt

        # time vector, interval, indices of observations
        varDA.fdvar.twind_obsInterval = varDA.fdvar.tw / (varDA.fdvar.nobstimes-1)
        varDA.fdvar.twind_obsTimes    = varDA.fdvar.twind[::varDA.fdvar.twind_obsInterval]
        varDA.fdvar.twind_obsIndex    = np.array(np.rint(varDA.fdvar.twind_obsTimes / model.dt), dtype=int)
        varDA.fdvar.twind_obs         = np.linspace(DA.t0,varDA.fdvar.twind_obsInterval, varDA.fdvar.twind_obsInterval +1) * model.dt

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, hybrid=DA.do_hybrid)
    if ( DA.do_hybrid ):
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), np.diag(H), np.diag(R), central_prior=xbc, central_posterior=xac, evratio=np.NaN, niters=np.NaN)
    else:
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), np.diag(H), np.diag(R), evratio=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        if ( DA.do_hybrid and varDA.update == 2 ):
            xs = advance_model(model, xt, varDA.fdvar.tfore, perfect=True)
            xt = xs[varDA.fdvar.ta,:].copy()
        else:
            xs = advance_model(model, xt, DA.tanal, perfect=True)
            xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
        ver = xt.copy()
        if ( DA.do_hybrid ):
            if   ( varDA.update == 1 ):
                ywin = y.copy()
            elif ( varDA.update == 2 ):
                ywin = np.zeros((varDA.fdvar.nobstimes,model.Ndof))
                for i in range(0,varDA.fdvar.nobstimes):
                    ywin[i,:] = np.dot(H,xs[varDA.fdvar.twind_obsIndex[i]+varDA.fdvar.tb,:] + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
                oind = np.where(varDA.fdvar.twind_obsIndex + varDA.fdvar.tb == varDA.fdvar.ta)
                if ( len(oind[0]) != 0 ): ywin[oind[0][0],:] = y.copy()

        if ( DA.do_hybrid ): # advance and update BOTH central and ensemble (mean and perturbations)

            if   ( varDA.update == 1 ):

                Xb = advance_ensemble(Xa, DA.tanal, model, perfect=False)
                Xbwin = Xb.copy()

                Xa, evratio = update_ensDA(Xb, y, R, H, ensDA, model)

            elif ( varDA.update == 2 ):

                Xb = advance_ensemble(Xa, varDA.fdvar.tbkgd, model, perfect=False)
                Xbwin = Xb.copy()

                for i in range(0,varDA.fdvar.nobstimes):

                    if ( varDA.fdvar.tb + varDA.fdvar.twind_obsIndex[i] > varDA.fdvar.ta ): break

                    Xa, evratio = update_ensDA(Xb, ywin[i,:], R, H, ensDA, model)

                    if ( varDA.fdvar.tb + varDA.fdvar.twind_obsIndex[i] < varDA.fdvar.ta ):
                        Xb = advance_ensemble(Xa, varDA.fdvar.twind_obs, model, perfect=False)

            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Be = np.cov(Xbwin, ddof=1)
            Bc = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * (Be*L)
            if ( varDA.precondition ):
                [U,S2,_] = np.linalg.svd(Bc, full_matrices=True, compute_uv=True)
                Bc = np.dot(U,np.diag(np.sqrt(S2)))

            # advance central analysis with the full nonlinear model
            xs = advance_model(model, xac, DA.tanal, perfect=False)
            xbc = xs[-1,:].copy()
            if   ( varDA.update == 1 ): xbcwin = xs[-1,            :].copy()
            elif ( varDA.update == 2 ): xbcwin = xs[varDA.fdvar.tb,:].copy()

            # update the central background
            xacwin, niters = update_varDA(xbcwin, Bc, ywin, R, H, varDA, model)

            # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
            if   ( varDA.update == 1 ):
                xac = xacwin.copy()
            elif ( varDA.update == 2 ):
                xs = advance_model(model, xacwin, varDA.fdvar.tanal, perfect=False)
                xac = xs[-1,:].copy()

        else: # advance and update ONLY ensemble (mean and perturbations)

            Xb = advance_ensemble(Xa, DA.tanal, model, perfect=False)
            Xa, evratio = update_ensDA(Xb, y, R, H, ensDA, model)

        # write diagnostics to disk before recentering
        if ( DA.do_hybrid ):
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, np.diag(H), np.diag(R), central_prior=xbc, central_posterior=xac, evratio=evratio, niters=niters)
        else:
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, np.diag(H), np.diag(R), evratio=evratio)

        # recenter ensemble about central analysis
        if ( DA.do_hybrid ):
            if ( DA.hybrid_rcnt ): Xa = np.transpose(np.transpose(Xa) - np.mean(Xa,axis=1) + xac)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
