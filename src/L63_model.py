#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_model.py - integrate the 1963 Lorenz model with different
#                ode solvers.
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy      as     np
from   lorenz     import L63, L63_tlm, plot_L63
from   plot_stats import plot_trace
from   scipy      import integrate
from   matplotlib import pyplot

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

t0 = 0.0
dt = 0.001

# setup from Miller et al., 1994
x0 = np.array([1.508870, -1.531271, 25.46091])
par  = np.array([10.0, 28.0, 8.0/3.0])
Ndof = 3

print 'spinning-up ON the attractor ...'

tf = 25.0
ts = np.arange(t0,tf+dt,dt)

xs = integrate.odeint(L63, x0, ts, (par,0.0))

#plot_L63(xs)
plot_trace(ver = np.transpose(xs), label=['x','y','z'], N=Ndof)

# let final state of previous integration be IC of next integration
x0 = xs[-1,:].copy()

tf = 0.05
ts = np.arange(t0,tf+dt,dt)

xs = integrate.odeint(L63, x0, ts, (par,0.0))
xsf = xs[-1,:].copy()

xp0 = np.random.randn(Ndof) * 1e-5

xsp = integrate.odeint(L63, x0+xp0, ts, (par,0.0))
xspf = xsp[-1,:].copy()

xp = integrate.odeint(L63_tlm, xp0, ts, (par, xs, ts, False))
xpf = xp[-1,:].copy()

print 'check TLM ...'
for j in range(0,Ndof):
    print 'j = %2d | Ratio = %f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])

xa0 = xpf.copy()
xa = integrate.odeint(L63_tlm, xa0, ts, (par, np.flipud(xs), ts, True))
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint ... %f' % (q2 -q1)

pyplot.show()
