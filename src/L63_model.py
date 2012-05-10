#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_model.py - integrate the 1963 Lorenz model,
#                test its TLM and adjoint
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import numpy         as     np
from   scipy         import integrate
from   matplotlib    import pyplot
from   plot_stats    import *
from   module_Lorenz import *
###############################################################

###############################################################
# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model      = type('',(),{})             # model Class
model.Name = 'L63'                      # model name
model.Ndof = 3                          # model degrees of freedom
model.Par  = [10.0, 28.0, 8.0/3.0]      # model parameters F, dF
model.dt   = 1.0e-3                     # model time-step

tf  = 0.25

IC          = type('',(),{})
IC.time     = None
IC.filename = ''
[x0,_] = get_IC(model,IC)

tol  = 1.0e-13
pert = 1.0e-4

print 'spinning-up ON the attractor ...'
print '--------------------------------'

ts = np.rint(np.linspace(0,1000*tf/model.dt,1000*tf/model.dt+1)) * model.dt
exec('xs = integrate.odeint(%s, x0, ts, (model.Par,0.0))' % (model.Name))
exec('plot_%s(xs)' % (model.Name))
x0 = xs[-1,:].copy()

ts = np.rint(np.linspace(0,tf/model.dt,tf/model.dt+1)) * model.dt

exec('xs = integrate.odeint(%s, x0, ts, (model.Par,0.0),rtol=tol,atol=tol)' % (model.Name))
xsf = xs[-1,:].copy()

xp0 = np.random.randn(model.Ndof) * pert

exec('xsp = integrate.odeint(%s, x0+xp0, ts, (model.Par,0.0),rtol=tol,atol=tol)' % (model.Name))
xspf = xsp[-1,:].copy()

exec('xp = integrate.odeint(%s_tlm, xp0, ts, (model.Par, xs, ts, False),rtol=tol,atol=tol)' % (model.Name))
xpf = xp[-1,:].copy()

print 'check TLM ..'
for j in range(0,model.Ndof):
    print 'j = %2d | Ratio = %14.13f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])
print '--------------------------------'

xa0 = xpf.copy()
exec('xa = integrate.odeint(%s_tlm, xa0, ts, (model.Par, np.flipud(xs), ts, True),rtol=tol,atol=tol)' % (model.Name))
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint .. %14.13f' % (q2-q1)
print '--------------------------------'

pyplot.show()
###############################################################
