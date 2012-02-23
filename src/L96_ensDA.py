#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_ensDA.py - cycle Ensemble DA on Lorenz & Emanuel 1998
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
import numpy      as     np
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   lorenz     import L96, plot_L96
from   ensDA      import *
from   plot_stats import *
###############################################################

###############################################################
global Ndof, F, dF, lab
global Q, H, R
global nassim, ntimes, dt, t0
global Eupdate, Nens, inflation, localization
global use_climo

Ndof = 40
F    = 8.0
dF   = 0.0
lab  = []
for j in range(0,Ndof): lab.append( 'x' + str(j+1) )

Q = np.eye(Ndof)*0.0            # model error variance (covariance model is white for now)
H = np.eye(Ndof)                # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(Ndof)*(4.0**2)       # observation error covariance

nassim = 160                    # no. of assimilation cycles
ntimes = 0.05                   # do assimilation every ntimes non-dimensional time units
dt     = 1.0e-4                 # time-step
t0     = 0.0                    # initial time

Eupdate      = 2                # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens         = 20               # number of ensemble members
localize     = True             # do localization
cov_cutoff   = 1.0              # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
localization = [localize, cov_cutoff]
infl_meth    = 1                # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac     = 1.01             # Depends on inflation method (see values in [] above)
inflation    = [infl_meth, infl_fac]

use_climo = False               # option to use climatological covariance (False = flow dependent)
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # initial setup from LE1998
    x0    = np.ones(Ndof) * F
    x0[0] = 1.001 * F

    # Make a copy of truth for plotting later
    xt    = x0.copy()
    truth = x0.copy()

    # populate initial ensemble analysis by perturbing true state
    [tmp, Xa] = np.meshgrid(np.ones(Nens),xt)
    pert = 0.001 * ( np.random.randn(Ndof,Nens) )
    Xa = Xa + pert
    Xb = Xa.copy()

    if ( use_climo ):
        print 'using climatological covariance ...'
        #data = io.loadmat('L96_climo_B.mat')
        #Bc   = data['B']copy
        nc = Dataset('L96_climo_B.nc4','r')
        Bc = nc.variables['B'][:]
        nc.close()

    print 'Cycling ON the attractor ...'

    ts = np.arange(t0,ntimes+dt,dt)     # time between assimilations

    # initialize arrays for statistics before cycling
    evstats = np.zeros(nassim) * np.NaN
    xbrmse  = np.zeros(nassim) * np.NaN
    xarmse  = np.zeros(nassim) * np.NaN
    xyrmse  = np.zeros(nassim) * np.NaN

    hist_ver       = np.zeros((Ndof,nassim)) * np.NaN
    hist_obs       = np.zeros((Ndof,nassim)) * np.NaN
    hist_xbm       = np.zeros((Ndof,nassim)) * np.NaN
    hist_xam       = np.zeros((Ndof,nassim)) * np.NaN
    hist_obs_truth = np.zeros((Ndof,(nassim+1)*(len(ts)-1)+1)) * np.NaN

    for k in range(0, nassim):

        print '========== assimilation time = %d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        xs = integrate.odeint(L96, xt, ts, (F,0.0))
        truth = np.vstack([truth, xs[1:,:]])
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance background ensemble with the full nonlinear model
        for m in range(0,Nens):
            xa = Xa[:,m].copy()
            xs = integrate.odeint(L96, xa, ts, (F+dF,0.0))
            Xb[:,m] = xs[-1,:].copy()

        # compute background ensemble mean and perturbations from the mean
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        # compute background error covariance; optionally, add model error covariance
        if ( use_climo ):
            B = Bc.copy()
        else:
            B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1) + Q

        # update ensemble (mean and perturbations)
        Xa, A, evstats[k] = update_ensDA(Xb, B, y, R, H, Eupdate=Eupdate, inflation=inflation, localization=localization)
        xam = np.mean(Xa,axis=1)

        # error statistics for ensemble mean
        xbrmse[k] = np.sqrt( np.sum( (ver - xbm)**2 ) / Ndof )
        xarmse[k] = np.sqrt( np.sum( (ver - xam)**2 ) / Ndof )
        xyrmse[k] = np.sqrt( np.sum( (ver -   y)**2 ) / Ndof )

        # history (for plotting)
        hist_ver[:,k] = ver
        hist_obs[:,k] = y
        hist_xbm[:,k] = xbm
        hist_xam[:,k] = xam
        hist_obs_truth[:,(k+1)*(len(ts)-1)+1] = y

        plot_L96(obs=y, ver=ver, xa=Xa, t=k+1, N=Ndof, figNum=1)
        pyplot.pause(0.1)

    # make some plots
    plot_trace(obs=hist_obs, ver=hist_ver, xb=hist_xbm, xa=hist_xam, label=lab, N=5)
    plot_rmse(xbrmse, xarmse, yscale='linear')
    plot_error_variance_stats(evstats)

    pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
