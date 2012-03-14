#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_stats.py - compute climatological covariance matrix for
#                the 1998 Lorenz & Emanuel system
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import os
import sys
import numpy         as     np
from   netCDF4       import Dataset
from   scipy         import integrate
from   module_Lorenz import L96

dt = 1.0e-4    # time-step
ts = 10.0      # time for spin-up  (50 days)
tf = 0.05      # time for forecast (6 hours)
Ne = 500       # no. of samples to estimate B
pscale = 0.001 # scale of perturbations to add

# initial setup from LE1998
Ndof  = 40
F     = 8.0
dF    = 0.0
x0    = np.ones(Ndof) * F
x0[0] = 1.001 * F

# get a state on the attractor
print 'spinning-up onto the attractor ...'
ts = np.arange(0.0,ts+dt,dt)       # how long to run onto the attractor
xs = integrate.odeint(L96, x0, ts, (F,0.0))

# use the end state as IC
xt = xs[-1,:]

# allocate space upfront
X = np.zeros((Ndof,Ne))

print 'running ON the attractor ...'

tf0 = np.arange(0.0,1*tf+dt,dt)
tf1 = np.arange(0.0,3*tf+dt,dt)
tf2 = np.arange(0.0,4*tf+dt,dt)

for i in range(0,Ne):
    xs = integrate.odeint(L96, xt, tf0, (F+dF,0.0))
    xt = xs[-1,:].copy()

    xs = integrate.odeint(L96, xt, tf1, (F+dF,0.0))
    x24 = xs[-1,:].copy()

    xs = integrate.odeint(L96, x24, tf2, (F+dF,0.0))
    x48 = xs[-1,:].copy()

    X[:,i] = x48 - x24

    xt = xt + pscale * np.random.randn(Ndof)

# compute climatological covariance matrix
B = np.dot(X,np.transpose(X)) / (Ne - 1)

# save B to disk for use with DA experiments
print 'save B to disk ...'
nc       = Dataset('L96_climo_B.nc4',mode='w',clobber=True,format='NETCDF4')
Dim      = nc.createDimension('xyz',Ndof)
Var      = nc.createVariable('B', 'f8', ('xyz','xyz',))
Var[:,:] = B
nc.close()
