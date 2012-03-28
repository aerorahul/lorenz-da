#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_model.py - integrate the 1998 Lorenz and Emanuel model,
#                test its TLM and adjoint
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy         as     np
from   scipy         import integrate
from   matplotlib    import pyplot
from   plot_stats    import *
from   module_Lorenz import *

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model      = type('',(),{})  # model Class
model.Name = 'L96'           # model name
model.Ndof = 40              # model degrees of freedom
model.Par  = [8.0, 0.4]      # model parameters F, dF

# initial setup from LE1998
x0    = np.ones(model.Ndof) * model.Par[0]
x0[0] = 1.001 * model.Par[0]

dt  = 1.0e-4
t0  = 0.0
tol = 1.0e-13
nf  = 4.0

print 'spinning-up ON the attractor ...'
print '--------------------------------'

tf = 25.0
ts = np.linspace(t0,tf,tf/dt+1.0,endpoint=True)

xs = integrate.odeint(L96, x0, ts, (model.Par[0],0.0))
plot_L96(ver=xs[-1,:],obs=xs[-1,:],t=tf,N=model.Ndof)

x0 = xs[-1,:].copy()

tf = 0.05 * nf
ts = np.linspace(t0,tf,tf/dt+1.0,endpoint=True)

xs = integrate.odeint(L96, x0, ts, (model.Par[0],0.0),rtol=tol,atol=tol)
xsf = xs[-1,:].copy()

xp0 = np.random.randn(model.Ndof) * 1.0e-4

xsp = integrate.odeint(L96, x0+xp0, ts, (model.Par[0],0.0),rtol=tol,atol=tol)
xspf = xsp[-1,:].copy()

xp = integrate.odeint(L96_tlm, xp0, ts, (model.Par[0], xs, ts, False),rtol=tol,atol=tol)
xpf = xp[-1,:].copy()

print 'check TLM ..'
for j in range(0,model.Ndof):
    print 'j = %2d | Ratio = %14.13f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])
print '--------------------------------'

xa0 = xpf.copy()
xa = integrate.odeint(L96_tlm, xa0, ts, (model.Par[0], np.flipud(xs), ts, True),rtol=tol,atol=tol)
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint .. %14.13f' % (q2 -q1)
print '--------------------------------'

pyplot.show()
