#!/usr/bin/env python3

###############################################################
# LXX_model.py - driver script for the Lorenz class models
#                and their respective TLM and adjoint
###############################################################

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from ruamel.yaml import YAML
from matplotlib import pyplot

from Lorenz import LorenzBase

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

parser = ArgumentParser(description='Test TLM and Adjoint for Lorenz class of models',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input', help='input yaml file containing the model configuration',
                    type=str, required=False, default='L96.yaml')
args = parser.parse_args()

yaml = YAML(typ='safe')
with open(args.input, 'r') as f:
    fullConf = yaml.load(f)

modelConf = fullConf.get('model', None)
if modelConf is None:
    raise NotImplementedError("Yaml file is missing model section")

model = LorenzBase.create(modelConf)

# Get initial conditions
#xt, x0 = model.getIC()
x0 = np.array([1.508870, -1.531271, 25.46091])

# How long to run (ncycles; each cycle is 6 hrs (0.05 time-units))
tf = 4 * 0.05

print('spinning-up ON the attractor ...')
print('--------------------------------')

ncycles = 250
ts = np.rint(np.linspace(0, ncycles * tf / model.dt,
                         int(ncycles * tf / model.dt) + 1)) * model.dt
xs = model.advance(x0, ts)
x0 = xs[-1, :].copy()
model.plot(ver=x0, obs=x0)
pyplot.show()

tol = 1.0e-13
pert = 1.0e-4

ts = np.rint(np.linspace(0, tf / model.dt, int(tf / model.dt) + 1)) * model.dt

xs = model.advance(x0, ts, rtol=tol, atol=tol)
xsf = xs[-1, :].copy()

xp0 = np.random.randn(model.Ndof) * pert

xsp = model.advance(x0 + xp0, ts, rtol=tol, atol=tol)
xspf = xsp[-1, :].copy()

xp = model.advance(xp0, ts, xtraj=xs,
                   ttraj=ts, adjoint=False, rtol=tol, atol=tol)
xpf = xp[-1, :].copy()

print('check TLM ...')
for j in range(0, model.Ndof):
    print('j = %2d | Ratio = %14.13f' % (j + 1, (xspf[j] - xsf[j]) / xpf[j]))
print('-------------')

xa0 = xpf.copy()
xa = model.advance(xa0, ts, xtraj=xs,
                   ttraj=ts, adjoint=True, rtol=tol, atol=tol)
xaf = xa[-1, :].copy()

q1 = np.dot(np.transpose(xpf), xpf)
q2 = np.dot(np.transpose(xaf), xp0)

print('check adjoint .. %14.13f' % (q2 - q1))
print('-------------')

pyplot.show()
