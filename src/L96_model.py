#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_model.py - integrate the 1998 Lorenz and Emanuel model
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy      as     np
from   lorenz     import L96, L96_tlm
from   plot_stats import plot_trace
from   scipy      import integrate
from   matplotlib import pyplot

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

dt = 1.0e-4
t0 = 0.0

tol = 1.0e-12

# initial setup from LE1998
Ndof  = 40
F     = 8.0
x0    = np.ones(Ndof) * F
x0[0] = 1.001 * F

label_list = []
for j in range(1,Ndof+1): label_list.append( 'x' + str(j) )

print 'spinning-up ON the attractor ...'

tf = 25.0
ts = np.arange(t0,tf+dt,dt)

xs = integrate.odeint(L96, x0, ts, (F,0.0))

plot_trace(ver = np.transpose(xs), label=label_list, N=3)

# let final state of previous integration be IC of next integration
x0 = xs[-1,:].copy()

tf = 0.05 * 4
ts = np.arange(t0,tf+dt,dt)

xs = integrate.odeint(L96, x0, ts, (F,0.0),rtol=tol,atol=tol)
xsf = xs[-1,:].copy()

xp0 = np.random.randn(Ndof) * 1.0e-4

xsp = integrate.odeint(L96, x0+xp0, ts, (F,0.0),rtol=tol,atol=tol)
xspf = xsp[-1,:].copy()

xp = integrate.odeint(L96_tlm, xp0, ts, (F, xs, ts, False),rtol=tol,atol=tol)
xpf = xp[-1,:].copy()

print 'check TLM ...'
for j in range(0,Ndof):
    print 'j = %2d | Ratio = %f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])

xa0 = xpf.copy()
xa = integrate.odeint(L96_tlm, xa0, ts, (F, np.flipud(xs), ts, True),rtol=tol,atol=tol)
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint ... %f' % (q2 -q1)

pyplot.show()
