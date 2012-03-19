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

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import os
import sys
import numpy         as     np
from   netCDF4       import Dataset
from   scipy         import integrate
from   module_Lorenz import L96
from   module_IO     import *
###############################################################

###############################################################
def main():

    method = 'EnKF' # choose method to create B: NMC / Climo / EnKF

    if ( (method == 'NMC') or (method == 'Climo') ):

        dt = 1.0e-4     # time-step
        tf = 0.05       # time for forecast (6 hours)
        ts = 50*4*tf    # time for spin-up  (50 days)
        Ne = 500        # no. of samples to estimate B

        # initial setup from LE1998
        Ndof  = 40
        F     = 8.0
        dF    = 0.0
        x0    = np.ones(Ndof) * F
        x0[0] = 1.001 * F

        if ( method == 'NMC' ):
            pscale = 1.0e-3         # scale of perturbations to add

    elif ( method == 'EnKF' ):

        # get the name of output diagnostic file to read
        [fname] = get_input_arguments()
        if ( not os.path.isfile(fname) ):
            print '%s does not exist' % fname
            sys.exit(1)

    if ( (method == 'NMC') or (method == 'Climo') ):

        # get a state on the attractor
        print 'spinning-up onto the attractor ...'
        ts = np.arange(0.0,ts+dt,dt)       # how long to run onto the attractor
        xs = integrate.odeint(L96, x0, ts, (F,0.0))

        # use the end state as IC
        xt = xs[-1,:]

        print 'Using the NMC method to create B'

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

    elif ( method == 'EnKF'):

        print 'Using the EnKF output to create B'

        try:
            nc   = Dataset(fname, mode='r', format='NETCDF4')
            Ndof = len(nc.dimensions['ndof'])
            Nens = len(nc.dimensions['ncopy'])
            Ntim = len(nc.dimensions['ntime'])
            Xb   = np.transpose(np.squeeze(nc.variables['prior'][:,]),(0,2,1))
            nc.close()
        except Exception as Instance:
            print 'Exception occurred during read of ' + fname
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

        print 'no. of samples ... %d' % Ntim

        Bi = np.zeros((Ntim,Ndof,Ndof))
        for i in range(0,Ntim):
            Bi[i,] = np.cov(np.squeeze(Xb[i,]),ddof=1)

        B = np.mean(Bi,axis=0)
        print np.diag(B)

    # save B to disk for use with DA experiments
    print 'save B to disk ...'
    nc       = Dataset('L96_climo_B_' + method +'.nc4',mode='w',clobber=True,format='NETCDF4')
    Dim      = nc.createDimension('xyz',Ndof)
    Var      = nc.createVariable('B', 'f8', ('xyz','xyz',))
    Var[:,:] = B
    nc.close()

    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
