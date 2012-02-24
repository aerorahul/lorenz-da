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
import numpy      as     np
from   lorenz     import L96
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   matplotlib import pyplot

dt = 1.0e-3   # time-step
ts = 100.0    # time for spin-up
tc = 1000.0   # time for climatology

# initial setup from LE1998
Ndof  = 40
F     = 8.0
x0    = np.ones(Ndof) * F
x0[0] = 1.001 * F

# get a state on the attractor
print 'spinning-up onto the attractor ...'
ts = np.arange(0.0,ts+dt,dt)       # how long to run onto the attractor
xs = integrate.odeint(L96, x0, ts, (F,0.0))

# use the end state to run ON the attractor
xt = xs[-1,:]

print 'running ON the attractor ...'
ts = np.arange(0.0,tc+dt,dt)
X = integrate.odeint(L96, xt, ts, (F,0.0))

nsamp = np.shape(X)[0]
print 'number of samples : %d' % nsamp

# calculate sample mean
xm = np.mean(X,axis=0)

# remove the sample mean from the sample
Xp = X - xm

# compute climatological covariance matrix
B = np.cov(np.transpose(Xp),ddof=1)

# save B to disk for use with DA experiments (both MatLAB and netCDF)
print 'save B to disk ...'
os.system('rm -f L96_climo_B.mat L96_climo_B.nc4')
data      = {}
data['B'] = B
io.savemat('L96_climo_B.mat',data)

nc       = Dataset('L96_climo_B.nc4',mode='w',clobber=True,format='NETCDF4')
Dim      = nc.createDimension('xyz',Ndof)
Var      = nc.createVariable('B', 'f8', ('xyz','xyz',))
Var[:,:] = B
nc.close()
