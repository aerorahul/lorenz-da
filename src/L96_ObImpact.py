#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_ObImpact.py - compare observation impact from ESA and
#                   adjoint methods using LE98 model
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
from   lorenz     import L96, L96_tlm, plot_L96
from   ensDA      import *
from   varDA      import *
from   plot_stats import *
###############################################################

###############################################################
global Ndof, F, dF, lab
global Q, H, R
global nassim, ntimes, dt, t0
global Eupdate, Nens, inflation, localization
global Vupdate, minimization
global hybrid_wght, do_hybrid
global nf, mxf

Ndof = 40
F    = 8.0
dF   = 0.0
lab  = []
for j in range(0,Ndof): lab.append( 'x' + str(j+1) )

Q = np.eye(Ndof)*0.0            # model error variance (covariance model is white for now)
H = np.eye(Ndof)                # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(Ndof)*(4.0**2)       # observation error covariance

nassim = 1000                   # no. of assimilation cycles
ntimes = 0.05                   # do assimilation every ntimes non-dimensional time units
dt     = 1.0e-4                 # time-step
t0     = 0.0                    # initial time

Eupdate      = 2                # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens         = 20               # number of ensemble members
localize     = True             # do localization
cov_cutoff   = 1.0              # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
localization = [localize, cov_cutoff]
infl_meth    = 1                # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac     = 1.02             # Depends on inflation method (see values in [] above)
inflation    = [infl_meth, infl_fac]

Vupdate = 1                     # variational-based DA method (1= 3Dvar; 2= 4Dvar)
maxiter = 1000                  # maximum iterations for minimization
alpha   = 4e-3                  # size of step in direction of normalized J
cg      = True                  # True = Use conjugate gradient; False = Perform line search
minimization = [maxiter, alpha, cg]

hybrid_wght = 0.0               # weight for hybrid (0.0= varDA; 1.0= ensDA)
do_hybrid   = True              # True= re-center ensemble about varDA, False= only ensDA

nf     = 4                      # extended forecast length : tf = nf * ntimes
mxf    = np.zeros(Ndof)
mxf[0] = 1.0                    # metric: single variable
mxf    = np.ones(Ndof)          # metric: sum of variables
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

    print 'load static climatological covariance ...'
    nc = Dataset('L96_climo_B.nc4','r')
    Bs = nc.variables['B'][:]
    nc.close()

    print 'Cycling ON the attractor ...'

    ts = np.arange(t0,ntimes+dt,dt)     # time between assimilations
    tf = np.arange(t0,nf*ntimes+dt,dt)  # extended forecast

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

    dJe = np.zeros(nassim) * np.NaN
    dJa = np.zeros(nassim) * np.NaN

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

        if ( do_hybrid ):
            # advance central analysis with the full nonlinear model
            xs = integrate.odeint(L96, xam, ts, (F+dF,0.0))
            xbc = xs[-1,:].copy()

        # compute background ensemble mean and perturbations from the mean
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        # compute background error covariance
        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1)

        # update ensemble (mean and perturbations)
        Xa, A, evstats[k] = update_ensDA(Xb, B, y, R, H, Eupdate=Eupdate, inflation=inflation, localization=localization)
        xam = np.mean(Xa,axis=1)
        Xap = np.transpose(np.transpose(Xa) - xam)

        if ( do_hybrid ):
            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Bc = (1.0 - hybrid_wght) * Bs + hybrid_wght * B

            # update the central trajectory
            xac, Ac, itstats[k] = update_varDA(xbc, Bc, y, R, H, Vupdate=Vupdate, minimization=minimization)

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

        fig1 = plot_L96(obs=y, ver=ver, xa=Xa, t=k+1, N=Ndof, figNum=1)
        pyplot.pause(0.1)

        # observation impact

        # advance analysis ensemble with the full nonlinear model
        Xf = np.zeros((Ndof,Nens))
        for m in range(0,Nens):
            xa = Xa[:,m].copy()
            xf = integrate.odeint(L96, xa, tf, (F+dF,0.0))
            Xf[:,m] = xf[-1,:].copy()

        # construct metric : J = (x^T)Wx ; dJ/dx = J_x = 2Wx ; choose W = I, x = xfmet, J_x = Jxf
        xfmet = np.transpose(mxf * np.transpose(Xf))
        J = np.diag(np.dot(np.transpose(xfmet), xfmet))

        Jp = J - np.mean(J,axis=0)
        Xo = Xap.copy()
        ye  = np.dot(H,Xb)
        mye = np.mean(ye,axis=1)
        dy = y - mye

        dJe[k] = np.dot(Jp,np.dot(np.transpose(np.dot(H,Xo)),np.dot(np.linalg.inv(R),dy))) / (Nens - 1)

        # advance analysis ensemble mean with the full nonlinear model
        xfm = integrate.odeint(L96, xam, tf, (F+dF,0.0))
        Jxf = mxf * 2 * xfm[-1,:]

        if ( do_hybrid ):
            dy = y - np.dot(H,xbc)

        # integrate the metric gradient from the end time to the initial time using the adjoint
        Jxa = integrate.odeint(L96_tlm, Jxf, tf, (F+dF,np.flipud(xfm),tf,True))
        Jxi = Jxa[-1,:].copy()
        dJa[k] = np.dot(Jxi,np.dot(A,np.dot(np.transpose(H),np.dot(np.linalg.inv(R),dy))))

    # make some plots
    fig2 = plot_trace(obs=hist_obs, ver=hist_ver, xb=hist_xbm, xa=hist_xam, label=lab, N=3)
    fig3 = plot_rmse(xbrmse, xarmse, yscale='linear')
    fig3.savefig('L96_RMSE.png',dpi=100,orientation='landscape',format='png')
    fig4 = plot_iteration_stats(itstats)
    fig4.savefig('L96_ItStats.png',dpi=100,orientation='landscape',format='png')
    fig5 = plot_error_variance_stats(evstats)
    fig5.savefig('L96_EVStats.png',dpi=100,orientation='landscape',format='png')
    fig6 = plot_ObImpact(dJa=dJa, dJe=dJe)
    fig6.savefig('L96_ObImpact.png',dpi=100,orientation='landscape',format='png')

    pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
