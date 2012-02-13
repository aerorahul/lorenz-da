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

dt = 0.001

# setup from Lorenz & Emanuel, 1998
F = 8.0
Ndof = 40
x0 = np.ones(Ndof) * F
x0[19] = x0[19] + 0.008

# get a state on the attractor
print 'spinning-up onto the attractor ...'
ts = np.arange(0.0,100.0,dt)       # how long to run onto the attractor
xs = integrate.odeint(L96, x0, ts, (F,0.0))

# use the end state to run ON the attractor
xt = xs[-1,:]

print 'running ON the attractor ...'
ts = np.arange(0.0,1000.0,dt)
xs = integrate.odeint(L96, xt, ts, (F,0.0))

X = xs.copy()
nsamp = np.shape(X)[0]
print 'number of samples : %d' % nsamp

# calculate sample mean
xm = np.mean(X,axis=0)
[Xm, tmp] = np.meshgrid(xm,np.ones(nsamp));

# remove the sample mean from the sample
Xp = np.transpose(X - Xm)

# compute climatological covariance matrix
B = np.dot(Xp,np.transpose(Xp)) / (nsamp - 1)

# option to save B to disk for use with DA experiments (both MatLAB and netCDF)
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

# plot the attractor

pyplot.show()
