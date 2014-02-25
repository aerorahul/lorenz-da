#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# LXX_stats.py - compute climatological covariances for the
#                1963 Lorenz and 1998 Lorenz & Emanuel systems
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
from   netCDF4       import Dataset
from   module_Lorenz import *
from   module_IO     import *
from   argparse      import ArgumentParser, ArgumentDefaultsHelpFormatter

###############################################################

###############################################################
def main():

    parser = ArgumentParser(description = 'Compute climatological covariances for LXX models', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--model',help='model name',type=str,choices=['L63','L96'],default='L96',required=False)
    parser.add_argument('-c','--covariances',help='covariance method',type=str,choices=['NMC','Climo','EnKF'],default='EnKF',required=False)
    parser.add_argument('-f','--filename',help='filename for EnKF method',type=str,required=True)
    args = parser.parse_args()

    method = args.covariances # choose method to create B: NMC / Climo / EnKF

    model = Lorenz()
    if   ( args.model == 'L63' ):
        Ndof = 3                          # model degrees of freedom
        Par  = [10.0, 28.0, 8.0/3.0]      # model parameters [sigma, rho, beta]
        dt   = 1.0e-4                     # model time-step
        tf   =  0.25                      # time for forecast (6 hours)
    elif ( args.model == 'L96' ):
        Ndof = 40                         # model degrees of freedom
        Par  = [8.0, 8.4]                 # model parameters F, F+dF
        dt   = 1.0e-4                     # model time-step
        tf   =  0.05                      # time for forecast (6 hours)
    model.init(Name=args.model,Ndof=Ndof,Par=Par,dt=dt)

    if ( (method == 'NMC') or (method == 'Climo') ):

        ts = 50*4*tf    # time for spin-up  (50 days)
        Ne = 500        # no. of samples to estimate B

        # initial setup
        IC          = type('',(),{})
        IC.time     = None
        IC.filename = ''
        [x0,_] = get_IC(model,IC)

        if ( method == 'NMC' ): pscale = 1.0e-3   # scale of perturbations to add

    elif ( method == 'EnKF' ):

        # get the name of EnKF output diagnostic file to read
        fname = args.filename

    if ( (method == 'NMC') or (method == 'Climo') ):

        # get a state on the attractor
        print 'spinning-up onto the attractor ...'
        ts = np.arange(0.0,ts+model.dt,model.dt)       # how long to run onto the attractor
        xs = model.advance(x0, ts, perfect=True)

        # use the end state as IC
        xt = xs[-1,:]

        print 'Using the NMC method to create B'

        # allocate space upfront
        X = np.zeros((model.Ndof,Ne))

        print 'running ON the attractor ...'

        tf0 = np.arange(0.0,1*tf+model.dt,model.dt)
        tf1 = np.arange(0.0,3*tf+model.dt,model.dt)
        tf2 = np.arange(0.0,4*tf+model.dt,model.dt)

        for i in range(0,Ne):
            xs = model.advance(xt, tf0, perfect=True)
            xt = xs[-1,:].copy()

            xs = model.advance(xt, tf1, perfect=False)
            x24 = xs[-1,:].copy()

            xs = model.advance(x24, tf2, perfect=False)
            x48 = xs[-1,:].copy()

            X[:,i] = x48 - x24

            xt = xt + pscale * np.random.randn(model.Ndof)

        # compute climatological covariance matrix
        B = np.dot(X,np.transpose(X)) / (Ne - 1)

    elif ( method == 'EnKF'):

        print 'Using the EnKF output to create B'
        [model_tmp, DA, ensDA, varDA] = read_diag_info(fname)
        _, Xb, _, _, _, _, _ = read_diag(fname, 0, end_time=DA.nassim)

        if ( (model.Name != model_tmp.Name) or (model.Ndof != model_tmp.Ndof) ):
            print 'mismatch between models, please verify'
            sys.exit(1)

        print 'no. of samples ... %d' % DA.nassim
        Ntim = DA.nassim
        offset = 500
        print 'removing first %d samples to account for spin-up ...' % offset

        Bi = np.zeros((Ntim-offset,model.Ndof,model.Ndof))
        for i in range(offset+0,Ntim): Bi[i-offset,] = np.cov(np.transpose(np.squeeze(Xb[i,])),ddof=1)

        B = np.mean(Bi,axis=0)

    # save B to disk for use with DA experiments
    print 'save B to disk ...'
    print np.diag(B)
    nc       = Dataset('L96_climo_B_' + method +'.nc4',mode='w',clobber=True,format='NETCDF4')
    Dim      = nc.createDimension('xyz',model.Ndof)
    Var      = nc.createVariable('B', 'f8', ('xyz','xyz',))
    Var[:,:] = B
    nc.close()
    print '... all done ...'

    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
