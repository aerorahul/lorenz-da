#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# LXX_model.py - driver script for the Lorenz 1963 and
#                1998 Lorenz and Emanuel model,
#                test their respective TLM and adjoint
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
import numpy         as     np
from   matplotlib    import pyplot
from   module_Lorenz import *
from   plot_stats    import *
from   argparse      import ArgumentParser, ArgumentDefaultsHelpFormatter

###############################################################

parser = ArgumentParser(description = 'Test TLM and Adjoint for LXX models', formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-m','--model',help='model name',type=str,required=False,choices=['L63','L96'],default='L96')
args = parser.parse_args()

###############################################################
# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model = Lorenz() # model Class
if   ( args.model == 'L63' ):
    Ndof = 3                          # model degrees of freedom
    Par  = [10.0, 28.0, 8.0/3.0]      # model parameters [sigma, rho, beta]
    dt   = 1.0e-3                     # model time-step
elif ( args.model == 'L96' ):
    Ndof = 40                         # model degrees of freedom
    Par  = [8.0, 8.4]                 # model parameters F, F+dF
    dt   = 1.0e-4                     # model time-step
model.init(Name=args.model,Ndof=Ndof,Par=Par,dt=dt)

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
xs = model.advance(x0, ts, perfect=True)
x0 = xs[-1,:].copy()

if   ( model.Name == 'L63' ): exec('plot_%s(att=xs)'                         % (model.Name            ))
elif ( model.Name == 'L96' ): exec('plot_%s(ver=xs[-1,:],obs=xs[-1,:],N=%d)' % (model.Name, model.Ndof))

ts = np.rint(np.linspace(0,tf/model.dt,tf/model.dt+1)) * model.dt

xs = model.advance(x0, ts, perfect=True, rtol=tol, atol=tol)
xsf = xs[-1,:].copy()

xp0 = np.random.randn(model.Ndof) * pert

xsp = model.advance(x0+xp0, ts, perfect=True, rtol=tol, atol=tol)
xspf = xsp[-1,:].copy()

xp = model.advance_tlm(xp0, ts, xs, ts, adjoint=False, perfect=True, rtol=tol, atol=tol)
xpf = xp[-1,:].copy()

print 'check TLM ..'
for j in range(0,model.Ndof):
    print 'j = %2d | Ratio = %14.13f' % (j+1, ( xspf[j] - xsf[j] ) / xpf[j])
print '--------------------------------'

xa0 = xpf.copy()
xa = model.advance_tlm(xa0, ts, xs, ts, adjoint=True, perfect=True, rtol=tol, atol=tol)
xaf = xa[-1,:].copy()

q1 = np.dot(np.transpose(xpf),xpf)
q2 = np.dot(np.transpose(xaf),xp0)

print 'check adjoint .. %14.13f' % (q2-q1)
print '--------------------------------'

pyplot.show()
###############################################################
