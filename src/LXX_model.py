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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from plot_stats import *
from module_Lorenz import *
from matplotlib import pyplot
import numpy as np
import sys
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################

###############################################################

parser = ArgumentParser(
    description='Test TLM and Adjoint for LXX models',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m',
    '--model',
    help='model name',
    type=str,
    required=False,
    choices=[
        'L63',
        'L96',
        'L96_2scale'],
    default='L96')
args = parser.parse_args()

###############################################################
# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model = Lorenz()  # model Class
if (args.model == 'L63'):
    # model parameters [sigma, rho, beta]
    Par = [10.0, 28.0, 8.0 / 3.0]
    Ndof = 3                                   # model degrees of freedom
    dt = 1.0e-3                              # model time-step
elif (args.model == 'L96'):
    Par = [8.0, 8.4]                          # model parameters [F, F+dF]
    Ndof = 40                                  # model degrees of freedom
    dt = 1.0e-4                              # model time-step
elif (args.model == 'L96_2scale'):
    # model parameters [F, F+dF, m, n, b, c, h]
    Par = [8.0, 8.4, 40, 4, 10.0, 10.0, 1.0]
    Ndof = 40 * (4 + 1)                            # model degrees of freedom
    dt = 1.0e-4                              # model time-step
    print('L96_2scale is still under development')
    print('You may experience dizziness!')
model.init(Name=args.model, Ndof=Ndof, Par=Par, dt=dt)

IC = type('', (), {})
IC.time = None
IC.filename = ''
[x0, _] = get_IC(model, IC)

tf = 4 * 0.05

tol = 1.0e-13
pert = 1.0e-4

print('spinning-up ON the attractor ...')
print('--------------------------------')

ts = np.linspace(0, 1000 * tf / model.dt,
                 int(1000 * tf / model.dt + 1)) * model.dt
xs = model.advance(x0, ts, perfect=True)
x0 = xs[-1, :].copy()

if (model.Name == 'L63'):
    exec('plot_%s(att=xs)' % (model.Name))
elif (model.Name == 'L96'):
    exec('plot_%s(ver=xs[-1,:],obs=xs[-1,:],N=%d)' % (model.Name, model.Ndof))
elif (model.Name == 'L96_2scale'):
    exec('plot_%s(ver=xs[-1,:],obs=xs[-1,:],N=%d)' % (model.Name, model.Ndof))

ts = np.linspace(0, tf / model.dt, int(tf / model.dt + 1)) * model.dt

xs = model.advance(x0, ts, perfect=True, rtol=tol, atol=tol)
xsf = xs[-1, :].copy()

xp0 = np.random.randn(model.Ndof) * pert

xsp = model.advance(x0 + xp0, ts, perfect=True, rtol=tol, atol=tol)
xspf = xsp[-1, :].copy()

xp = model.advance_tlm(
    xp0,
    ts,
    xs,
    ts,
    adjoint=False,
    perfect=True,
    rtol=tol,
    atol=tol)
xpf = xp[-1, :].copy()

print('check TLM ..')
for j in range(0, model.Ndof):
    print(('j = %2d | Ratio = %14.13f' % (j + 1, (xspf[j] - xsf[j]) / xpf[j])))
print('--------------------------------')

xa0 = xpf.copy()
xa = model.advance_tlm(
    xa0,
    ts,
    xs,
    ts,
    adjoint=True,
    perfect=True,
    rtol=tol,
    atol=tol)
xaf = xa[-1, :].copy()

q1 = np.dot(np.transpose(xpf), xpf)
q2 = np.dot(np.transpose(xaf), xp0)

print(('check adjoint .. %14.13f' % (q2 - q1)))
print('--------------------------------')

pyplot.show()
###############################################################
