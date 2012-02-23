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
import numpy      as     np
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   lorenz     import L96, plot_L96
from   ensDA      import *
from   varDA      import *
from   plot_stats import *
###############################################################

###############################################################
global Ndof, F, dF, lab
global Q, H, R
global nassim, ntimes, dt, t0
global Eupdate, Nens, infl, infl_fac, loc, cov_cutoff
global Vupdate, maxiter, alpha, cg
global use_climo

Ndof = 40
F    = 8.0
dF   = 0.4
lab  = []
for j in range(0,Ndof): lab.append( 'x' + str(j+1) )

Q = np.eye(Ndof)*0.0            # model error variance (covariance model is white for now)
H = np.eye(Ndof)                # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(Ndof)*(4.0**2)       # observation error covariance

nassim = 16                     # no. of assimilation cycles
ntimes = 0.05                   # do assimilation every ntimes non-dimensional time units
dt     = 1.0e-4                 # time-step
t0     = 0.0                    # initial time

Eupdate    = 2                  # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens       = 20                 # number of ensemble members
loc        = True               # localization
cov_cutoff = 1.0                # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
infl       = 1                  # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac   = 1.02               # Depends on inflation method (see values in [] above)

Vupdate = 1                     # DA method (1= 3Dvar; 2= 4Dvar)
maxiter = 1000                  # maximum iterations
alpha   = 4e-3                  # size of step in direction of normalized J
cg      = True                  # True = Use conjugate gradient; False = Perform line search

hybrid_wght = 0.0               # weight for hybrid (0= varDA; 1= ensDA)
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
    xam = np.mean(Xa,axis=1)
    Xb = Xa.copy()
    xbm = xam.copy()

    print 'using climatological covariance ...'
    nc = Dataset('L96_climo_B.nc4','r')
    Bs = nc.variables['B'][:]
    nc.close()

    print 'Cycling ON the attractor ...'

    ts = np.arange(t0,ntimes+dt,dt)     # time between assimilations

    # initialize arrays for statistics before cycling
    evstats = np.zeros(nassim) * np.NaN
    itstats = np.zeros(nassim) * np.NaN
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

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,Nens):
            xa = Xa[:,m].copy()
            xs = integrate.odeint(L96, xa, ts, (F+dF,0.0))
            Xb[:,m] = xs[-1,:].copy()

        # advance central analysis with the full nonlinear model
        xs = integrate.odeint(L96, xam, ts, (F+dF,0.0))
        xbc = xs[-1,:].copy()

        # compute background ensemble mean and perturbations from the mean
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        # compute background error covariance
        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1)

        # update ensemble (mean and perturbations)
        xam, Xap, Xa, A, evstats[k] = update_ensDA(xbm, Xbp, Xb, B, y, H, R)

        # blend covariance from flow-dependent (ensemble) and static (climatology)
        Bc = (1.0 - hybrid_wght) * Bs + hybrid_wght * B

        # update the central trajectory
        xac, Ac, itstats[k] = update_varDA(xbc, Bc, y, R, H)

        # replace ensemble mean analysis with central analysis
        xam = xac.copy()
        Xa = np.transpose(xam + np.transpose(Xap))

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
    plot_iteration_stats(itstats)
    plot_error_variance_stats(evstats)

    pyplot.show()
###############################################################

###############################################################
def update_ensDA(xbm, Xbp, Xb, B, y, H, R):

    Nobs = np.shape(y)[0]

    innov  = np.zeros(Nobs)
    totvar = np.zeros(Nobs)

    temp_ens = Xb.copy()

    for ob in range(0, Nobs):

        ye = temp_ens[ob,:]

        if   ( Eupdate == 0 ): # no assimilation
            obs_inc, innov[ob], totvar[ob] = np.zeros(Ndof), 0.0, 0.0

        elif ( Eupdate == 1 ): # update using the EnKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnKF(y[ob], R[ob,ob], ye)

        elif ( Eupdate == 2 ): # update using the EnSRF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnSRF(y[ob], R[ob,ob], ye)

        elif ( Eupdate == 3 ): # update using the EAKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EAKF(y[ob], R[ob,ob], ye)

        else:
            print 'invalid update algorithm ...'
            sys.exit(2)

        for i in range(0,Ndof):
            state_inc = state_increment(obs_inc, temp_ens[i,:], ye)

            # localization
            if ( loc ):
                dist = np.abs( ob - i ) / Ndof
                if ( dist > 0.5 ): dist = 1.0 - dist
                cov_factor = compute_cov_factor(dist, cov_cutoff)
            else:
                cov_factor = 1.0

            temp_ens[i,:] = temp_ens[i,:] + state_inc * cov_factor

    Xa = temp_ens.copy()

    # compute analysis mean and perturbations
    xam = np.mean(Xa,axis=1)
    Xap = np.transpose(np.transpose(Xa) - xam)

    # inflation
    if   ( infl == 1 ): # multiplicative inflation
        Xap = infl_fac * Xap

    elif ( infl == 2 ): # additive white model error (zero-mean, infl_fac-spread)
        Xap = Xap + infl_fac * np.random.randn(Ndof,Nens)

    elif ( infl == 3 ): # covariance relaxation (Zhang, Snyder)
        Xap = Xbp * infl_fac + Xap * (1.0 - infl_fac)

    elif ( infl == 4 ): # posterior spread restoration (Whitaker & Hammill)
        xbs = np.std(Xb,axis=1)
        xas = np.std(Xa,axis=1)
        for i in np.arange(0,Ndof):
            Xap[i,:] =  np.sqrt((infl_fac * (xbs[i] - xas[dof])/xas[i]) + 1.0) * Xap[i,:]

    else:
        print 'invalid inflation algorithm ...'
        sys.exit(2)

    # add inflated perturbations back to analysis mean
    Xa = np.transpose(np.transpose(Xap) + xam)

    # compute analysis error covariance matrix
    A = np.dot(Xap,np.transpose(Xap)) / (Nens - 1)

    # check for filter divergence
    error_variance_ratio = np.sum(innov**2) / np.sum(totvar)
    if not ( 0.5 < error_variance_ratio < 2.0 ):
        print 'FILTER DIVERGENCE : ERROR / TOTAL VARIANCE = %f' % (error_variance_ratio)
        #break

    return xam, Xap, Xa, A, error_variance_ratio
###############################################################

###############################################################
def update_varDA(xb, B, y, R, H):
    if ( Vupdate == 1 ):
        [xa, A, niters] = ThreeDvar(xb, B, y, R, H, maxiter=maxiter, alpha=alpha, cg=cg)

    elif ( Vupdate == 2 ):
        [xa, A, niters] = FourDvar(xb, B, y, R, H, maxiter=maxiter, alpha=alpha, cg=cg)

    else:
        print 'invalid update algorithm ...'
        sys.exit(2)

    return xa, A, niters
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
