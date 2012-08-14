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
    check_varDA(varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xb = Xa.copy()
    if ( DA.do_hybrid ):
        xac = np.mean(Xa,axis=1)
        xbc = np.mean(Xb,axis=1)
        if ( fdvar ): Xbb = Xa.copy()

    # load climatological covariance once and for all ...
    if ( DA.do_hybrid ): Bs = read_clim_cov(model)

    # construct localization matrix once and for all ...
    L = localization_operator(model,varDA.localization)

    if ( fdvar ):
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
        if ( fdvar ):
            xs = advance_model(model, xt, varDA.fdvar.tfore, perfect=True)
            xt = xs[varDA.fdvar.ta,:].copy()
        else:
            xs = advance_model(model, xt, DA.tanal, perfect=True)
            xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
        ver = xt.copy()
        if ( fdvar ):
            ywin = np.zeros((varDA.fdvar.nobstimes,model.Ndof))
            for i in range(0,varDA.fdvar.nobstimes):
                ywin[i,:] = np.dot(H,xs[varDA.fdvar.twind_obsIndex[i]+varDA.fdvar.tb,:] + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
            oind = np.where(varDA.fdvar.twind_obsIndex + varDA.fdvar.tb == varDA.fdvar.ta)
            if ( len(oind[0]) != 0 ): y = ywin[oind[0][0],:].copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            xs = advance_model(model, xa, DA.tanal, perfect=False)
            Xb[:,m] = xs[-1,:].copy()
            if ( (DA.do_hybrid) and (fdvar) ): Xbb[:,m] = xs[varDA.fdvar.tb,:].copy()

        # compute background error covariance from the ensemble
        if ( DA.do_hybrid ):
            if ( fdvar ): Be = np.cov(Xbb, ddof=1)
            else:         Be = np.cov(Xb,  ddof=1)

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA)

        if ( DA.do_hybrid ):
            # advance central analysis with the full nonlinear model
            xs = advance_model(model, xac, DA.tanal, perfect=False)
            xbc = xs[-1,:].copy()
            if ( fdvar ): xbcwin = xs[varDA.fdvar.tb,:].copy()

            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Bc = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * (Be*L)

            # update the central background
            if ( fdvar ): xacwin, niters = update_varDA(xbcwin, Bc, ywin, R, H, varDA, model=model)
            else:         xac,    niters = update_varDA(xbc,    Bc, y,    R, H, varDA, model=model)

            # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
            if ( fdvar ):
                xs = advance_model(model, xacwin, varDA.fdvar.tanal, perfect=False)
                xac = xs[-1,:].copy()

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
