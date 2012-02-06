#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_stats.py - compute covariance matrix for the 1963 Lorenz
#                attractor
#
# created : Oct 2011 : Rahul Mahajan : GMAO / GSFC / NASA
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import os
import numpy      as     np
from   lorenz     import L63
from   netCDF4    import Dataset
from   scipy      import integrate, io
from   matplotlib import pyplot

dt = 0.01 # time-step

# control parameter settings for Lorenz 63
par = np.array([10.0, 28.0, 8.0/3.0])

# get a state on the attractor
print 'running onto the attractor ...'
x0 = np.array([10.0, 20.0, 30.0])  # initial conditions
ts = np.arange(0.0,100.0,dt)       # how long to run onto the attractor
xs = integrate.odeint(L63, x0, ts, (par,0.0))

# use the end state to run ON the attractor
xt = xs[-1,:]

print 'running ON the attractor ...'
ts = np.arange(0.0,1000.0,dt)
xs = integrate.odeint(L63, xt, ts, (par,0.0))

X = xs
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
os.system('rm -f L63_climo_B.mat L63_climo_B.nc4')
data      = {}
data['B'] = B
io.savemat('L63_climo_B.mat',data)

nc       = Dataset('L63_climo_B.nc4',mode='w',clobber=True,format='NETCDF4')
Dim      = nc.createDimension('xyz',3)
Var      = nc.createVariable('B', 'f8', ('xyz','xyz',))
Var[:,:] = B
nc.close()

fig = pyplot.figure()
pyplot.clf()
pyplot.plot(xs[:,0], xs[:,2], color='gray', linewidth=1)
pyplot.xlabel('X',fontweight='bold',fontsize=12)
pyplot.ylabel('Z',fontweight='bold',fontsize=12)
pyplot.plot(xs[50:155,0], xs[50:155,2], 'go-', linewidth=1)
pyplot.title('Lorenz attractor',fontweight='bold',fontsize=14)
pyplot.show()
