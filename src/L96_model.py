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

model      = type('',(),{})  # model Class
model.Name = 'L96'           # model name
model.Ndof = 40              # model degrees of freedom
model.Par  = [8.0, 8.4]      # model parameters F, dF
model.dt   = 1.0e-4          # model time-step

# initial setup from LE1998
IC          = type('',(),{})
IC.time     = None
IC.filename = ''
[x0,_] = get_IC(model,IC)

tf  = 4*0.05

tol  = 1.0e-13
pert = 1.0e-4

print 'spinning-up ON the attractor ...'
print '--------------------------------'

ts = np.rint(np.linspace(0,1000*tf/model.dt,1000*tf/model.dt+1)) * model.dt
xs = advance_model(model, x0, ts, perfect=True)
exec('plot_%s(ver=xs[-1,:],obs=xs[-1,:],t=25,N=%d)' % (model.Name, model.Ndof))
x0 = xs[-1,:].copy()

ts = np.rint(np.linspace(0,tf/model.dt,tf/model.dt+1)) * model.dt

xs = advance_model(model, x0, ts, perfect=True, rtol=tol, atol=tol)
xsf = xs[-1,:].copy()

xp0 = np.random.randn(model.Ndof) * pert

xsp = advance_model(model, x0+xp0, ts, perfect=True, rtol=tol, atol=tol)
xspf = xsp[-1,:].copy()

xp = advance_model_tlm(model, xp0, ts, xs, ts, adjoint=False, perfect=True, rtol=tol, atol=tol)
xpf = xp[-1,:].copy()

print 'check TLM ..'
for j in range(0,model.Ndof):
    print 'j = %2d | Ratio = %14.13f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])
print '--------------------------------'

xa0 = xpf.copy()
xa = advance_model_tlm(model, xa0, ts, xs, ts, adjoint=True, perfect=True, rtol=tol, atol=tol)
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint .. %14.13f' % (q2-q1)
print '--------------------------------'

pyplot.show()
###############################################################
