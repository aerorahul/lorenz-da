#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_hybDA.py - cycle Hybrid DA on the 1963 Lorenz attractor
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
from   ensDA      import PerturbedObs, Potter, EnKF
from   varDA      import ThreeDvar, FourDvar
from   lorenz     import L63, plot_L63
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   plot_stats import plot_trace, plot_abs_error, plot_abs_error_var
###############################################################

###############################################################
global Ndof, par, lab
global Q, H, R
global nassim, ntimes, dt
global Eupdate, Nens, infl, infl_fac
global Vupdate, maxiter, alpha, cg
global hybrid_wght

# settings for Lorenz 63
Ndof = 3
par  = np.array([10.0, 28.0, 8.0/3.0])
lab  = ['x', 'y', 'z']

Q         = np.eye(Ndof)*1e-3   # model error variance (covariance model is white for now)
H         = np.eye(Ndof)        # obs operator ( eye(3) gives identity obs )
R         = np.eye(Ndof)*1e-2   # observation error covariance

nassim    = 160                 # no. of assimilation cycles
ntimes    = 0.25                # do assimilation every ntimes non-dimensional time units
dt        = 0.01                # time-step

Eupdate  = 3                    # ens. DA method (1= Perturbed Obs; 2= Potter; 3= EnKF)
Nens     = 50                   # number of ensemble members
infl     = 1                    # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac = 1.01                 # Depends on inflation method (see values in [] above)

Vupdate = 1                     # var. DA method (1= 3Dvar; 2= 4Dvar)
maxiter = 100                   # maximum iterations
alpha   = 4e-3                  # size of step in direction of normalized J
cg      = True                  # True = Use conjugate gradient; False = Perform line search

hybrid_wght = 0.0               # weight for hybrid (0= varDA; 1= ensDA)
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

    # initial conditions from Miller et al., 1994
    #xs = np.array([1.508870, -1.531271, 25.46091])
    #xt = np.array([1.508870, -1.531271, 25.46091])

    # Make a copy of truth for plotting later
    #truth = xs.copy()
    truth = xt.copy()

    # populate initial ensemble analysis by perturbing true state
    [tmp, Xa] = np.meshgrid(np.ones(Nens),xt)
    pert = 1e-1 * np.random.randn(Ndof,Nens)
    Xa  = Xa + pert
    xam = np.mean(Xa,axis=1)
    Xb  = Xa.copy()
    xbm = xam.copy()

    print 'load static climatological covariance ...'
    nc = Dataset('L63_climo_B.nc4','r')
    Bs = nc.variables['B'][:]
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

        # advance analysis ensemble with the full nonlinear model
        for n in range(0,Nens):
            xa = Xa[:,n].copy()
            xs = integrate.odeint(L63, xa, ts, (par,0.0))
            Xb[:,n] = xs[-1,:].copy()

        # advance central analysis with the full nonlinear model
        xs = integrate.odeint(L63, xam, ts, (par,0.0))
        xbc = xs[-1,:].copy()

        # compute background ensemble mean and perturbations
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        # compute background error covariance matrix
#        if ( infl ): # square-root filter
#            Xbp = infl_fac * Xbp
#            # additive zero-mean white model error
#            Xbp = Xbp + np.dot(Q,np.random.randn(Ndof,Nens))
#        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1) + Q
        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1)

        # update ensemble (mean and perturbations)
        [xam, Xap, Xa, A] = update_ensDA(xbm, Xbp, Xb, B, y, R, H)

        # blend covariance from flow-dependent (ensemble) and static (climatology)
        Bc = (1.0 - hybrid_wght) * Bs + hybrid_wght * B

        # update the central trajectory
        [xac, Ac] = update_varDA(xbc, Bc, y, R, H)

        # replace ensemble mean analysis with central analysis
        xam = xac.copy()
        Xa = np.transpose(xam + np.transpose(Xap))

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

    if ( Eupdate == 1 ):   # update using perturbed observations
        Xa  = PerturbedObs(Xb, B, y, H, R)
        xam = np.mean(Xa,axis=1)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xap = Xa - Xam

    elif ( Eupdate == 2 ): # update using the Potter algorithm
        [xam, Xap] = Potter(xbm, Xbp, y, H, R)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap

    elif ( Eupdate == 3 ): # update using the EnKF algorithm
        loc = np.ones((Nobs,Ndof)) # this does no localization
        [xam, Xap] = EnKF(xbm, Xbp, y, H, R, loc)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap

    else:
        print 'invalid ensemble update algorithm ...'
        sys.exit(2)

    # Must inflate if using EnKF flavors
    if ( Eupdate > 1 ):

        if   ( infl == 1 ): # multiplicative inflation
            Xap = infl_fac * Xap

        elif ( infl == 2 ): # additive zero-mean white model error
            Xap = Xap + infl_fac * np.random.randn(Ndof,Nens)

        elif ( infl == 3 ): # covariance relaxation (Zhang, Snyder)
            Xap = Xbp * infl_fac + Xap * (1 - infl_fac)

        elif ( infl == 4 ): # posterior spread restoration (Whitaker & Hammill)
            xbs = np.std(Xb,axis=1)
            xas = np.std(Xa,axis=1)
            for dof in np.arange(0,Ndof):
                Xap[dof,:] =  np.sqrt((infl_fac * (xbs[dof] - xas[dof])/xas[dof]) + 1) * Xap[dof,:]

        elif ( infl == 5 ): # adaptive (Anderson)
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
def update_varDA(xb, B, y, R, H):

    if ( Vupdate == 1 ):
        [xa, A] = ThreeDvar(xb, B, y, R, H, maxiter=maxiter, alpha=alpha, cg=True)

    elif ( Vupdate == 2 ):
        [xa, A] = FourDvar(xb, B, y, R, H, maxiter=maxiter, alpha=alpha, cg=True)

    else:
        print 'invalid variational update algorithm ...'
        sys.exit(2)

    return xa, A
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
