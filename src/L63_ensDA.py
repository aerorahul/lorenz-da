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

import sys
import numpy      as     np
from   ensDA      import PerturbedObs, Potter, EnKF
from   lorenz     import L63
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   scipy      import integrate, io

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

# number of degrees of freedom in the Lorenz 63 system
Ndof = 3

update    = 3                   # DA method (1= Perturbed Obs; 2= Potter; 3= EnKF; 4= EAKF; 5= ETKF)
tavg_obs  = False               # time-averaged observations
q         = 0.001               # model error variance (covariance model is white for now)
R         = np.eye(Ndof)*1e-2   # observation error covariance
nassim    = 160                # no. of assimilation cycles
ntimes    = 0.25                # do assimilation every ntimes non-dimensional time units
dt        = 0.01                # time-step
use_climo = False               # option to use climatological covariance (False = flow dependent)
Nens      = 50                  # number of ensemble members
inflation = 1                   # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac  = 1.01                # Depends on inflation method (see values in [] above)
H         = np.eye(Ndof)        # obs operator ( eye(3) gives identity obs )

plot_error_statistics = True # True = Plot error statistics; False = Don't

# control parameter settings for Lorenz 63
par = np.array([10.0, 28.0, 8.0/3.0])

if ( update == 1 ):
    print 'running perturbed observations ...'
    inflation = 0
elif ( update == 2 ):
    print 'running the potter algorithm ...'
elif ( update == 3 ):
    print 'running the enkf algorithm ...'
elif ( update == 4 ):
    print 'running the eakf algorithm ...'
    print 'eakf algorithm not implemented yet ...'
    sys.exit(2)
elif ( update == 5 ):
    print 'running the etkf algorithm ...'
    print 'etkf algorithm not implemented yet ...'
    sys.exit(2)
else:
    print 'invalid update code specified'
    sys.exit(2)

# get a state on the attractor
print 'running onto the attractor ...'
x0 = np.array([10.0, 20.0, 30.0]) # initial conditions
ts = np.arange(0,100,0.02)        # how long to run on the attractor
xs = integrate.odeint(L63, x0, ts, (par,0.0))

# IC for truth taken from last time:
Xt = xs[-1,:].copy()

# Make a copy of truth for plotting later
truth = Xt.copy()
truth = xs.copy()

# populate initial ensemble analysis by perturbing true state
[tmp, Xa] = np.meshgrid(np.ones(Nens),Xt)
pert = 1e-1 * np.random.randn(Ndof,Nens)
Xa = Xa + pert

if ( use_climo ):
    print 'using climatological covariance ...'
    #data = io.loadmat('L63_climo_B.mat')
    #B    = data['B']copy
    nc = Dataset('L63_climo_B.nc4','r')
    B  = nc.variables['B'][:]
    nc.close()
    Bc = B.copy()

print 'Cycling ON the attractor ...'

ts = np.arange(0,ntimes,dt)     # how long to run on the attractor

# initialize arrays for statistics
xbe  = np.zeros((Ndof,nassim))
xae  = np.zeros((Ndof,nassim))
xye  = np.zeros((Ndof,nassim))
xbev = np.zeros((Ndof,nassim))
xaev = np.zeros((Ndof,nassim))

hist_ver = np.zeros((Ndof,nassim))
hist_obs = np.zeros((Ndof,nassim))
hist_xbm = np.zeros((Ndof,nassim))
hist_xam = np.zeros((Ndof,nassim))

for k in range(0, nassim):

    print '=== assimilation time = %d ================== ' % (k+1)

    Xb = Xa.copy()
    if ( tavg_obs ):
        Xbtpert = Xa.copy()

    # advance truth with the full nonlinear model
    xs = integrate.odeint(L63, Xt, ts, (par,0.0))
    truth = np.vstack([truth, xs[1:,:]])
    Xt = xs[-1,:].copy()

    # new observations from noise about truth; set verification values
    if ( tavg_obs ):
        Y = np.dot(H,np.mean(xs,axis=0)) + np.diag(np.diag(np.random.randn(Ndof))*np.sqrt(R))
        Ver = np.mean(xs,axis=0)
    else:
        Y = np.dot(H,Xt) + np.diag(np.diag(np.random.randn(Ndof))*np.sqrt(R))
        Ver = Xt.copy()

    Nobs = np.shape(Y)[0]

    # advance background ensemble with the full nonlinear model
    for n in range(0,Nens):

        xb = Xb[:,n].copy()
        xs = integrate.odeint(L63, xb, ts, (par,0.0))

        if ( tavg_obs ):
            Xb[:,n]  = np.mean(xs,axis=0) # time-mean ensemble
            Xbtpert[:,n] = xs[-1,:].copy() - Xb[:,n]
        else:
            Xb[:,n] = xs[-1,:].copy()

    xbm = np.mean(Xb,axis=1)

    if ( use_climo ):
        B = Bc.copy()
    else:
        # remove the ensemble mean xbm from ensemble estimate
        [tmp, Xbm] = np.meshgrid(np.ones(Nens),xbm)
        Xp = Xb - Xbm
        Xbp = Xp.copy()
        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1) + q*np.eye(Ndof)
#        if ( inflation ): # square-root filter
#            Xbp = inflation * Xp
#            # additive zero-mean white model error
#            Xbp = Xbp + q*np.random.randn(Ndof,Nens)
#        else: # perturbed obs.
#            Xbp = Xp.copy()
#        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1) + q*np.eye(Ndof)

    # update step for ensemble mean (xam), perturbations from the mean (Xap),
    # and the full state (Xa = Xam + Xap)
    if ( update == 1 ):   # update using perturbed observations
        Xa  = PerturbedObs(Xb, B, Y, R)
        xam = np.mean(Xa,axis=1)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xap = Xa - Xam
    elif ( update == 2 ): # update using the Potter algorithm
        [xam, Xap] = Potter(xbm, Xbp, Y, H, R)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap
    elif ( update == 3 ): # update using the EnKF algorithm
        loc = np.ones((Nobs,Ndof)) # this does no localization
        [xam, Xap] = EnKF(xbm, Xbp, Y, H, R, loc)
        [tmp, Xam] = np.meshgrid(np.ones(Nens),xam)
        Xa = Xam + Xap
    else:
        print 'EAKF, ETKF have not been implemented yet'
        sys.exit(2)

    if ( update > 1 ): # Must inflate if using EnKF flavors
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
            print 'Adaptive inflation is not implemented yet'
            sys.exit(2)
        else:
            print 'Unknown inflation method specified'
            sys.exit(2)
        Xa = Xam + Xap

    A = np.dot(Xap,np.transpose(Xap)) / (Nens - 1)

    print 'trace of B and A: %7.4f  %7.4f' % ( np.trace(B), np.trace(A) )

    # error statistics for ensemble mean
    xbe[:,k]  = xbm - Ver
    xae[:,k]  = xam - Ver
    xye[:,k]  = Y   - Ver
    xbev[:,k] = np.diag(B)
    xaev[:,k] = np.diag(A)

    # check for filter divergence
    if ( np.abs(xae[1,k]) > 10 and np.abs(xae[2,k]) > 10 ):
        print 'filter divergence'
        sys.exit(2)

    # history (for plotting)
    hist_ver[:,k] = Ver
    hist_obs[:,k] = Y
    hist_xbm[:,k] = xbm
    hist_xam[:,k] = xam

    # if time-averaged observations, add back the perturbations from the time-mean
    if ( tavg_obs ):
        Xa = Xa + Xbtpert

lab = ['x', 'y', 'z']

# absolute error in the ensemble mean for each dimension
fig = pyplot.figure(1)
pyplot.clf()
for k in range(0,Ndof):
    pyplot.subplot(Ndof,1,k+1)
    pyplot.hold(True)
    pyplot.plot(np.abs(xbe[k,:]),'b-',label='background',linewidth=2)
    pyplot.plot(np.abs(xae[k,:]),'r-',label='analysis',linewidth=2)
    pyplot.ylabel(lab[k],fontweight='bold',fontsize=12)
    strb = 'mean background : %5.4f +/- %5.4f' % (np.mean(np.abs(xbe[k,1:])), np.std(np.abs(xbe[k,1:]),ddof=1))
    stra = 'mean analysis : %5.4f +/- %5.4f' % (np.mean(np.abs(xae[k,1:])), np.std(np.abs(xae[k,1:]),ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    yoff = 0.9 * yl[1]
    pyplot.text(0,yoff,strb,fontsize=10)
    yoff = 0.8 * yl[1]
    pyplot.text(0,yoff,stra,fontsize=10)
    pyplot.hold(False)
    if ( k == 0 ):
        pyplot.legend(loc=1)
        pyplot.title('Absolute Error in Mean',fontweight='bold',fontsize=14)
    if ( k == Ndof-1 ):
        pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)

# background and analysis error variance
fig = pyplot.figure(2)
pyplot.clf()
for k in range(0,Ndof):
    pyplot.subplot(Ndof,1,k+1)
    pyplot.hold(True)
    pyplot.plot(xbev[k,2:],'b-',label='background',linewidth=2)
    pyplot.plot(xaev[k,2:],'r-',label='analysis',linewidth=2)
    pyplot.ylabel(lab[k],fontweight='bold',fontsize=12)
    strb = 'mean background : %5.4f +/- %5.4f' % (np.mean(xbev[k,2:]), np.std(xbev[k,2:],ddof=1))
    stra = 'mean analysis : %5.4f +/- %5.4f' % (np.mean(xaev[k,2:]), np.std(xaev[k,2:],ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    yoff = 0.9 * yl[1]
    pyplot.text(0,yoff,strb,fontsize=10)
    yoff = 0.8 * yl[1]
    pyplot.text(0,yoff,stra,fontsize=10)
    pyplot.hold(False)
    if ( k == 0 ):
        pyplot.legend(loc=1)
        pyplot.title('Ensemble Kalman Filter Error Variance ',fontweight='bold',fontsize=14)
    if ( k == Ndof-1 ):
        pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)

# time traces of the history variables
fig = pyplot.figure(3)
pyplot.clf()
for k in range(0,Ndof):
    pyplot.subplot(Ndof,1,k+1)
    pyplot.hold(True)
    pyplot.plot(hist_obs[k,:],'ro',label='observation')
    pyplot.plot(hist_ver[k,:],'k-',label='truth')
    pyplot.plot(hist_xbm[k,:],'c-',label='background')
    pyplot.plot(hist_xam[k,:],'b-',label='analysis')
    pyplot.ylabel(lab[k],fontweight='bold',fontsize=12)
    pyplot.hold(False)
    if ( k == 0 ):
        pyplot.legend(loc=0,ncol=2)
        pyplot.title('Time trace',fontweight='bold',fontsize=14)
    if ( k == Ndof-1 ):
        pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)

# Truth for the entire assimilation cycle
fig = pyplot.figure(4)
pyplot.clf()
pyplot.hold(True)
pyplot.plot(truth[:,0],truth[:,2],color='gray',linewidth=1)
pyplot.plot(truth[351:420,0],truth[351:420,2],'ro',linewidth=2)
pyplot.plot(truth[350,0],truth[350,2],'go',linewidth=2,markersize=10.0)
pyplot.plot(truth[421,0],truth[421,2],'go',linewidth=2,markersize=5.0)
pyplot.xlabel('X',fontweight='bold',fontsize=12)
pyplot.ylabel('Z',fontweight='bold',fontsize=12)
pyplot.title('Lorenz attractor',fontweight='bold',fontsize=14)

pyplot.show()
