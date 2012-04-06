#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_hybDA.py - cycle Hybrid DA on Lorenz & Emanuel 1998
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
from   scipy         import integrate, io
from   matplotlib    import pyplot
from   netCDF4       import Dataset
from   module_Lorenz import *
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
global model
global Q, H, R
global DA, ensDA, varDA
global diag_file
global restart

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 0.4]         # model parameters F, dF
model.dt   = 1.0e-4             # model time-step

Q = np.eye(model.Ndof)*0.0      # model error variance (covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

DA             = type('', (), {}) # data assimilation Class
DA.nassim      = 2000             # no. of assimilation cycles
DA.ntimes      = 0.05             # do assimilation every ntimes non-dimensional time units
DA.t0          = 0.0              # initial time
DA.do_hybrid   = True             # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
DA.hybrid_wght = 0.0              # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
DA.hybrid_rcnt = False            # True= re-center ensemble about varDA, False= free ensDA

ensDA              = type('', (), {})  # ensemble data assimilation Class
ensDA.inflation    = type('', (), {})  # inflation Class
ensDA.localization = type('', (), {})  # localization Class
ensDA.update                  = 2      # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 30     # number of ensemble members
ensDA.inflation.infl_meth     = 1      # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                       # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
ensDA.inflation.infl_fac      = 1.06   # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = True   # do localization
ensDA.localization.cov_cutoff = 1.0    # normalized covariance cutoff = cutoff / ( 2*normalized_dist)

varDA                      = type('', (), {}) # variational data assimilation Class
varDA.minimization         = type('', (), {}) # minimization Class
varDA.update               = 1                # variational-based DA method (1 = 3Dvar; 2= 4Dvar)
varDA.minimization.maxiter = 1000             # maximum iterations for minimization
varDA.minimization.alpha   = 4e-4             # size of step in direction of normalized J
varDA.minimization.cg      = True             # True = Use conjugate gradient; False = Perform line search
varDA.minimization.tol     = 1e-4             # tolerance to end the variational minimization iteration

if ( (varDA.update == 2) or (varDA.update == 4) ): fdvar = True
else:                                              fdvar = False

if ( fdvar ):
    varDA.fdvar                = type('',(),{}) # 4DVar class
    varDA.fdvar.maxouter       = 1              # no. of outer loops for 4DVar
    varDA.fdvar.window         = DA.ntimes      # length of the 4Dvar assimilation window
    varDA.fdvar.offset         = 0.5            # time offset: forecast from analysis to background time
    varDA.fdvar.nobstimes      = 11             # no. of evenly spaced obs. times in the window

# name and attributes of/in the output diagnostic file
diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = model.Name + '_hybDA_diag.nc4'
diag_file.attributes = {'F'           : str(model.Par[0]),
                        'dF'          : str(model.Par[1]),
                        'dt'          : str(model.dt),
                        'ntimes'      : str(DA.ntimes),
                        'do_hybrid'   : str(int(DA.do_hybrid)),
                        'hybrid_wght' : str(DA.hybrid_wght),
                        'hybrid_rcnt' : str(int(DA.hybrid_rcnt)),
                        'Eupdate'     : str(ensDA.update),
                        'infl_meth'   : str(ensDA.inflation.infl_meth),
                        'infl_fac'    : str(ensDA.inflation.infl_fac),
                        'localize'    : str(int(ensDA.localization.localize)),
                        'cov_cutoff'  : str(ensDA.localization.cov_cutoff),
                        'Vupdate'     : str(varDA.update),
                        'maxiter'     : str(varDA.minimization.maxiter),
                        'alpha'       : str(varDA.minimization.alpha),
                        'cg'          : str(int(varDA.minimization.cg)),
                        'tol'         : str(varDA.minimization.tol)}
if ( fdvar ):
    diag_file.attributes.update({'offset'    : str(varDA.fdvar.offset),
                                 'window'    : str(varDA.fdvar.window),
                                 'nobstimes' : str(int(varDA.fdvar.nobstimes)),
                                 'maxouter'  : str(int(varDA.fdvar.maxouter))})

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = None              # None == default | -N...-1 0 1...N
restart.filename = ''
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble and variational data assimilation options
    check_ensDA(ensDA)
    check_varDA(varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xb = Xa.copy()
    if ( DA.do_hybrid ):
        xac = np.mean(Xa,axis=1)
        xbc = np.mean(Xb,axis=1)
        if ( fdvar ): Xbb = Xa.copy()

    if ( DA.do_hybrid ):
        print 'load climatological covariance ...'
        nc = Dataset(model.Name + '_climo_B.nc4','r')
        Bs = nc.variables['B'][:]
        nc.close()

    if ( fdvar ):
        # check length of assimilation window
        if ( varDA.fdvar.offset * DA.ntimes + varDA.fdvar.window - DA.ntimes < 0.0 ):
            print 'assimilation window is too short'
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
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), central_prior=xbc, central_posterior=xac, evratio=np.NaN, niters=np.NaN)
    else:
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        if ( fdvar ):
            exec('xs = integrate.odeint(%s, xt, varDA.fdvar.tfore, (%f,0.0))' % (model.Name, model.Par[0]))
            xt = xs[varDA.fdvar.ta,:].copy()
        else:
            exec('xs = integrate.odeint(%s, xt, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]))
            xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()
        if ( fdvar ):
            ywin = np.zeros((varDA.fdvar.nobstimes,model.Ndof))
            for i in range(0,varDA.fdvar.nobstimes):
                ywin[i,:] = np.dot(H,xs[varDA.fdvar.twind_obsIndex[i]+varDA.fdvar.tb,:]) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            exec('xs = integrate.odeint(%s, xa, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
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
            if ( fdvar ):
                # step to the beginning of the assimilation window (varDA.fdvar.tbkgd)
                exec('xs = integrate.odeint(%s, xac, varDA.fdvar.tbkgd, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
            else:
                # step to the next assimilation time (DA.tanal)
                exec('xs = integrate.odeint(%s, xac, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
            xbc = xs[-1,:].copy()

            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Bc = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * Be

            # update the central background
            if ( fdvar ): xac, Ac, niters = update_varDA(xbc, Bc, ywin, R, H, varDA, model=model)
            else:         xac, Ac, niters = update_varDA(xbc, Bc, y,    R, H, varDA)

        # write diagnostics to disk before recentering
        if ( DA.do_hybrid ):
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), central_prior=xbc, central_posterior=xac, evratio=evratio, niters=niters)
        else:
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio=evratio)

        # recenter ensemble about central analysis
        if ( DA.do_hybrid ):
            if ( DA.hybrid_rcnt ): Xa = np.transpose(np.transpose(Xa) - np.mean(Xa,axis=1) + xac)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
