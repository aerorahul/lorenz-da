#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_ensDA.py - cycle Ensemble DA on the 1963 Lorenz attractor
#
# created : Oct 2011 : Rahul Mahajan : GMAO / GSFC / NASA
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import sys
import numpy      as     np
from   ensDA      import PerturbedObs, Potter, EnKF
from   lorenz     import L63, plot_L63
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   plot_stats import plot_trace, plot_abs_error, plot_abs_error_var
###############################################################

###############################################################
global Ndof, par, lab
global q, H, R
global nassim, ntimes, dt
global update, Nens, inflation, infl_fac
global use_climo

# settings for Lorenz 63
Ndof = 3
par  = np.array([10.0, 28.0, 8.0/3.0])
lab  = ['x', 'y', 'z']

q         = 0.001               # model error variance (covariance model is white for now)
H         = np.eye(Ndof)        # obs operator ( eye(3) gives identity obs )
R         = np.eye(Ndof)*1e-2   # observation error covariance

nassim    = 10                 # no. of assimilation cycles
ntimes    = 0.25                # do assimilation every ntimes non-dimensional time units
dt        = 0.01                # time-step

update    = 3                   # DA method (1= Perturbed Obs; 2= Potter; 3= EnKF; 4= EAKF; 5= ETKF)
Nens      = 50                  # number of ensemble members
inflation = 1                   # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac  = 1.01                # Depends on inflation method (see values in [] above)

use_climo = False               # option to use climatological covariance (False = flow dependent)
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # get a state on the attractor
    print 'running onto the attractor ...'
    x0 = np.array([10.0, 20.0, 30.0]) # initial conditions
    ts = np.arange(0.0,100.0,2*dt)    # how long to run on the attractor
    xs = integrate.odeint(L63, x0, ts, (par,0.0))

    # IC for truth taken from last time:
    xt = xs[-1,:].copy()

    # Make a copy of truth for plotting later
    truth = xt.copy()
    truth = xs.copy()

    # populate initial ensemble analysis by perturbing true state
    [tmp, Xa] = np.meshgrid(np.ones(Nens),xt)
    pert = 1e-1 * np.random.randn(Ndof,Nens)
    Xa = Xa + pert
    Xb = Xa.copy()

    if ( use_climo ):
        print 'using climatological covariance ...'
        #data = io.loadmat('L63_climo_B.mat')
        #Bc   = data['B']copy
        nc = Dataset('L63_climo_B.nc4','r')
        Bc = nc.variables['B'][:]
        nc.close()

    print 'Cycling ON the attractor ...'

    # initialize arrays for statistics before cycling
    xbe  = np.zeros((Ndof,nassim))
    xae  = np.zeros((Ndof,nassim))
    xye  = np.zeros((Ndof,nassim))
    xbev = np.zeros((Ndof,nassim))
    xaev = np.zeros((Ndof,nassim))

    hist_ver = np.zeros((Ndof,nassim))
    hist_obs = np.zeros((Ndof,nassim))
    hist_xbm = np.zeros((Ndof,nassim))
    hist_xam = np.zeros((Ndof,nassim))

    ts = np.arange(0,ntimes,dt)     # time between assimilations

    for k in range(0, nassim):

        print '========== assimilation time = %d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        xs = integrate.odeint(L63, xt, ts, (par,0.0))
        truth = np.vstack([truth, xs[1:,:]])
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.diag(np.diag(np.random.randn(Ndof))*np.sqrt(R))
        ver = xt.copy()

        # advance background ensemble with the full nonlinear model
        for n in range(0,Nens):
            xa = Xa[:,n].copy()
            xs = integrate.odeint(L63, xa, ts, (par,0.0))
            Xb[:,n] = xs[-1,:].copy()

        # compute background ensemble mean and perturbations from the mean
        xbm = np.mean(Xb,axis=1)
        [tmp, Xbm] = np.meshgrid(np.ones(Nens),xbm)
        Xbp = Xb - Xbm

        if ( use_climo ):
            B = Bc.copy()
        else:
#            if ( inflation ): # square-root filter
#                Xbp = inflation * Xbp
#                # additive zero-mean white model error
#                Xbp = Xbp + q*np.random.randn(Ndof,Nens)
#            B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1) + q*np.eye(Ndof)
            B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1)

        # update step
        [xam, Xap, Xa, A] = update_ensDA(xbm, Xbp, Xb, B, y, R, H)

        # error statistics for ensemble mean
        xbe[:,k]  = xbm - ver
        xae[:,k]  = xam - ver
        xye[:,k]  = y   - ver
        xbev[:,k] = np.diag(B)
        xaev[:,k] = np.diag(A)

        # history (for plotting)
        hist_ver[:,k] = ver
        hist_obs[:,k] = y
        hist_xbm[:,k] = xbm
        hist_xam[:,k] = xam

        # check for filter divergence
        if ( np.abs(xae[1,k]) > 10 and np.abs(xae[2,k]) > 10 ):
            print 'filter divergence'
            sys.exit(2)

    # make some plots
    plot_L63(truth,segment=xs)
    plot_trace(hist_obs, hist_ver, hist_xbm, hist_xam, label=lab, N=Ndof)
    plot_abs_error(xbe,xae,label=lab,N=Ndof)
    plot_abs_error_var(xbev,xaev,label=lab,N=Ndof)

    pyplot.show()
###############################################################

###############################################################
def update_ensDA(xbm, Xbp, Xb, B, y, R, H):

    Nobs = np.shape(y)[0]

    if ( update == 1 ):   # update using perturbed observations
        Xa  = PerturbedObs(Xb, B, y, H, R)
        xam = np.mean(Xa,axis=1)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xap = Xa - Xam

    elif ( update == 2 ): # update using the Potter algorithm
        [xam, Xap] = Potter(xbm, Xbp, y, H, R)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap

    elif ( update == 3 ): # update using the EnKF algorithm
        loc = np.ones((Nobs,Ndof)) # this does no localization
        [xam, Xap] = EnKF(xbm, Xbp, y, H, R, loc)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap

    else:
        print 'invalid update algorithm ...'
        sys.exit(2)

    # Must inflate if using EnKF flavors
    if ( update > 1 ):

        if   ( inflation == 1 ): # multiplicative inflation
            Xap = infl_fac * Xap

        elif ( inflation == 2 ): # additive zero-mean white model error
            Xap = Xap + infl_fac * np.random.randn(Ndof,Nens)

        elif ( inflation == 3 ): # covariance relaxation (Zhang, Snyder)
            Xap = Xbp * infl_fac + Xap * (1 - infl_fac)

        elif ( inflation == 4 ): # posterior spread restoration (Whitaker & Hammill)
            xbs = np.std(Xb,axis=1)
            xas = np.std(Xa,axis=1)
            for dof in np.arange(0,Ndof):
                Xap[dof,:] =  np.sqrt((infl_fac * (xbs[dof] - xas[dof])/xas[dof]) + 1) * Xap[dof,:]

        elif ( inflation == 5 ): # adaptive (Anderson)
            print 'adaptive inflation is not implemented yet'
            sys.exit(2)

        else:
            print 'invalid inflation algorithm ...'
            sys.exit(2)

        # add inflated perturbation back to analysis mean
        Xa = Xam + Xap

    # compute analysis error covariance matrix
    A = np.dot(Xap,np.transpose(Xap)) / (Nens - 1)

    print 'trace of B and A: %7.4f  %7.4f' % ( np.trace(B), np.trace(A) )

    return xam, Xap, Xa, A
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
