#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_adjoint_sensitivity.py - compute adjoint sensitivity for
#                              the 1963 Lorenz attractor
#
# created : Oct 2011 : Rahul Mahajan : GMAO / GSFC / NASA
###############################################################

from matplotlib import pyplot
from lorenz import L63, L63_tlm
from scipy import integrate, io
from netCDF4 import Dataset
import numpy as np
import sys
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"


# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

# number of degrees of freedom in the Lorenz 63 system
Ndof = 3

ts = 20.0         # how long to spin-up
tf = 4 * 0.25       # length of the forecast (0.25 => 6 hr forecast)
dt = 0.01         # time-step

# sensitivity gradient at forecast time
sxf = np.array([1.0, 0.0, 0.0])  # single variable
# sxf = np.array([1.0, 1.0, 1.0])  # sum of variables

# plot x v/s idim plots ( eg. 1 gives x-y plot, 2 gives x-z plot [most
# common] )
idim = 2

# control parameter settings for Lorenz 63
par = np.array([10.0, 28.0, 8.0 / 3.0])

# initial conditions
x0 = np.array([10.0, 20.0, 30.0])

# get a state on the attractor
print('running onto the attractor ...')
# how long to run (spin-up) on the attractor
ts = np.arange(0.0, ts + dt, dt)
xs = integrate.odeint(L63, x0, ts, (par, 0.0))

# IC for truth taken from last time:
xt = xs[-1, :].copy()

# get a bunch of points on the attractor for plotting
tplot = np.arange(0.0, 100.0 + 0.001, 0.001)
xplot = integrate.odeint(L63, xt, tplot, (par, 0.0))

# now make a control forecast
tsave = np.arange(0.0, tf + dt, dt)
xsave = integrate.odeint(L63, xt, tsave, (par, 0.0))

# end state
xf = xsave[-1, :].copy()

# we get the sensitivity by two methods:
# 1. "brute force": compute the propagator by unit impulses, one at a time for each dof
# 2. a single integration of the adjoint model

tint = np.arange(0.0, tf + dt, dt)

# 1. run the TLM to get M
sxp = np.array([1.0, 0.0, 0.0])
M1 = integrate.odeint(L63_tlm, sxp, tint, (par, xsave, tsave, False))
sxp = np.array([0.0, 1.0, 0.0])
M2 = integrate.odeint(L63_tlm, sxp, tint, (par, xsave, tsave, False))
sxp = np.array([0.0, 0.0, 1.0])
M3 = integrate.odeint(L63_tlm, sxp, tint, (par, xsave, tsave, False))

# construct propagator M
M = np.transpose(np.array([M1[-1, :], M2[-1, :], M3[-1, :]]))

# initial condition sensitivity : dJ/dx_0 = M^T dJ/dx_t
sxi = np.dot(np.transpose(M), sxf)

# 2. direct integration of the adjoint model from the end time to the initial time.
# note that the order of control trajectory is reversed in xsave, but the length of time in
# tint is the same (do not reverse the order in tint!)
axs = integrate.odeint(
    L63_tlm,
    sxf,
    tint,
    (par,
     np.flipud(xsave),
     tsave,
     True))

# present a check on the two methods
print()
print(
    'initial sensitivity gradient from M^T                 : %7.4f %7.4f %7.4f' %
     (sxi[0], sxi[1], sxi[2]))
print('initial sensitivity gradient from adjoint integration : %7.4f %7.4f %7.4f' %
      (axs[-1, 0], axs[-1, 1], axs[-1, 2]))

# Gradient check for the TLM and its adjoint:
tol = 1.0e-13  # convergence tolerance for scipy.integrate.odeint
sxi_tlm = integrate.odeint(
    L63_tlm,
    sxi,
    tint,
    (par,
     xsave,
     tsave,
     False),
    rtol=tol,
    atol=tol)
z = sxi_tlm[-1, :].copy()
zTz = np.dot(np.transpose(z), z)

axs = integrate.odeint(
    L63_tlm,
    z,
    tint,
    (par,
     np.flipud(xsave),
     tsave,
     True),
    rtol=tol,
    atol=tol)
z0 = axs[-1, :].copy()
z0Tsxi = np.dot(np.transpose(z0), sxi)

print()
print('Gradient check result : ' + str(zTz - z0Tsxi))

# plot the attractor,  control trajectory, and sensitivity gradient vector
fig = pyplot.figure(1)
pyplot.clf()
pyplot.hold(True)
pyplot.plot(xplot[:, 0], xplot[:, idim], color='gray', linewidth=1)
pyplot.plot(xsave[:, 0], xsave[:, idim], 'ro', linewidth=1)
pyplot.xlim((-25, 25))
pyplot.ylim((-5, 55))

vs = 10.0

# initial time sensitivity gradient vector
pyplot.plot(np.array([xsave[0, 0], xsave[0, 0] +
                      vs *
                      sxi[0]]), np.array([xsave[0, idim], xsave[0, idim] +
                                          vs *
                                          sxi[idim]]), 'g-', linewidth=4)

# forecast time sensitivity gradient vector
pyplot.plot(np.array([xsave[-1, 0], xsave[-1, 0] + vs * sxf[0]]), np.array(
    [xsave[-1, idim], xsave[-1, idim] + vs * sxf[idim]]), 'g-', linewidth=2)

pyplot.xlabel('X', fontweight='bold', fontsize=12)
pyplot.ylabel('modeled Z', fontweight='bold', fontsize=12)
pyplot.title(
    'Lorenz attractor, Control Trajectory, and Sensitivity Gradient Vector',
    fontweight='bold',
    fontsize=14)

# perturb the initial condition with the gradient and check sensitivity prediction
# xp = 0.1 * np.random.randn(3)  # random perturbation
xp = 0.1 * sxi                  # perturbation in the direction of sensitivity
pert = xt + xp
pxsave = integrate.odeint(L63, pert, tint, (par, 0.0))
px = pxsave - xsave

# add a plot of the perturbation vector
vs = vs / 0.1
for k in range(0, np.shape(px)[0], 4):
    pyplot.plot(np.array([xsave[k, 0], xsave[k, 0] +
                          vs *
                          px[k, 0]]), np.array([xsave[k, idim], xsave[k, idim] +
                                                vs *
                                                px[k, idim]]), 'b-', linewidth=2)

# check TLM against the non-linear integration
perttlm = np.dot(M, xp)

print()
print('TLM perturbation solution      : %7.4f %7.4f %7.4f' %
      (perttlm[0], perttlm[1], perttlm[2]))
print('non-linear difference solution : %7.4f %7.4f %7.4f' %
      (px[-1, 0], px[-1, 1], px[-1, 2]))

# predicted and modeled (actual) change in the forecast metric
Jc = xf[0].copy()              # single variable metric - Control
Jp = pxsave[-1, 0].copy()       # single variable metric - Perturbation
# Jc = sum(xf).copy()            # sum metric - Control
# Jp = sum(pxsave[-1,:]).copy()  # sum metric - Perturbation
dJp = np.dot(np.transpose(sxi), xp)    # predicted change
dJm = Jp - Jc                          # modeled (actual) change

print()
print('predicted change in metric : %7.4f' % (dJp))
print('  modeled change in metric : %7.4f' % (dJm))

xl = pyplot.get(pyplot.gca(), 'xlim')
yl = pyplot.get(pyplot.gca(), 'ylim')
pyplot.text(0.9 * xl[0], -4, 'control trajectory', color='red')
pyplot.text(0.9 * xl[0], -2,
            'adjoint sensitivity gradient vector', color='green')
pyplot.text(0.9 * xl[0], 0, 'perturbed state vector', color='blue')

pyplot.text(
    0.9 *
    xl[0],
    0.9 *
    yl[1],
    'predicted change in metric : %7.4f' %
    (dJp))
pyplot.text(
    0.9 *
    xl[0],
    0.9 *
    yl[1] -
    2,
    'modeled change in metric   : %7.4f' %
    (dJm))

# now test as a function of perturbation amplitude
amp_min = -5.0
amp_max = 5.0
amp_step = 0.1
dJp = np.zeros((len(np.arange(amp_min, amp_max + amp_step, amp_step)), 1))
dJm = dJp.copy()
k = -1
for amp in np.arange(amp_min, amp_max + amp_step, amp_step):
    k = k + 1
    xp = amp * sxi
    pert = xt + xp
    pxsave = integrate.odeint(L63, pert, tint, (par, 0.0))
    px = pxsave - xsave
    Jc = xf[0].copy()
    Jp = pxsave[-1, 0].copy()
    dJm[k] = Jp - Jc
    dJp[k] = np.dot(np.transpose(sxi), xp)

fig = pyplot.figure(2)
pyplot.clf()
pyplot.hold(True)
maxval = np.max([np.max(np.abs(dJp)), np.max(np.abs(dJm))])
maxval = 1.2 * maxval
pyplot.xlim((-maxval, maxval))
pyplot.ylim((-maxval, maxval))
xl = pyplot.get(pyplot.gca(), 'xlim')
yl = pyplot.get(pyplot.gca(), 'ylim')
pyplot.plot(np.array([xl[0], xl[1]]), np.array([yl[0], yl[1]]), linewidth=2.0)
pyplot.plot(dJp, dJm, 'ko', markersize=3.0)
pyplot.xlabel('predicted dJ', fontweight='bold', fontsize=12)
pyplot.ylabel('modeled dJ', fontweight='bold', fontsize=12)
pyplot.title(
    'Lorenz 1963 Adjoint Sensitivity Check as a Function of IC Amplitude',
    fontweight='bold',
    fontsize=14)

pyplot.show()
