#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# proc_L96.py - process the diagnostics written by L96_???DA.py
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
from   matplotlib    import pyplot
from   netCDF4       import Dataset
from   module_Lorenz import *
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
def main():

    # get the name of output diagnostic file to read, start index for statistics, index to plot
    # state
    [_,fname,sStat,ePlot] = get_input_arguments()
    if ( not os.path.isfile(fname) ):
        print '%s does not exist' % fname
        sys.exit(1)

    if ( sStat <= -1 ): sStat = 100

    # read dimensions and necessary attributes from the diagnostic file
    try:
        nc = Dataset(fname, mode='r', format='NETCDF4')
        ndof   = len(nc.dimensions['ndof'])
        nassim = len(nc.dimensions['ntime'])
        nobs   = len(nc.dimensions['nobs'])

        if 'ncopy' in nc.dimensions:
            nens = len(nc.dimensions['ncopy'])
        else:
            nens = 0

        if 'do_hybrid' in nc.ncattrs():
            do_hybrid = nc.do_hybrid
        else:
            do_hybrid = False

        ntimes = nc.ntimes
        dt     = nc.dt
        F      = nc.F
        dF     = nc.dF

        nc.close()
    except Exception as Instance:
        print 'Exception occurred during read of ' + fname
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    # print some info so the user knows the script is doing something
    print 'no. of assimilation cycles = %d' % nassim

    # read the diag file
    try:
        nc = Dataset(fname, mode='r', format='NETCDF4')

        xt = np.squeeze(nc.variables['truth'][:,])
        Xb = np.squeeze(nc.variables['prior'][:,])
        Xa = np.squeeze(nc.variables['posterior'][:,])
        y  = np.squeeze(nc.variables['obs'][:,])
        if ( do_hybrid ):
            Xb      = np.transpose(Xb, (0,2,1))
            Xa      = np.transpose(Xa, (0,2,1))
            xbm     = np.mean(Xb, axis=2)
            xam     = np.mean(Xa, axis=2)
            xbc     = np.squeeze(nc.variables['central_prior'][:,])
            xac     = np.squeeze(nc.variables['central_posterior'][:,])
            niters  = np.squeeze(nc.variables['niters'][:])
            evratio = np.squeeze(nc.variables['evratio'][:])
        else:
            if ( nens == 0 ):
                xbm    = Xb.copy()
                xam    = Xa.copy()
                niters = np.squeeze(nc.variables['niters'][:])
            else:
                Xb      = np.transpose(Xb, (0,2,1))
                Xa      = np.transpose(Xa, (0,2,1))
                xbm     = np.mean(Xb, axis=2)
                xam     = np.mean(Xa, axis=2)
                evratio = np.squeeze(nc.variables['evratio'][:])

        nc.close()
    except Exception as Instance:
        print 'Exception occurred during read of ' + fname
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    # compute RMSE in prior, posterior and observations
    xbrmse = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / ndof )
    xarmse = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / ndof )
    xyrmse = np.sqrt( np.sum( (xt -   y)**2          ) / ndof )

    # plot the last state
    fig = plot_L96(obs=y[ePlot,], ver=xt[ePlot,], xb=Xb[ePlot,], xa=Xa[ePlot,], t=ePlot, N=ndof)

    # plot the RMSE
    fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear')

    # plot the last state and RMSE for central state
    if ( do_hybrid ):
        xbrmse = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / ndof )
        xarmse = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / ndof )
        xyrmse = np.sqrt( np.sum( (xt -   y)**2          ) / ndof )
        fig = plot_L96(obs=y[ePlot,], ver=xt[ePlot,], xb=xbc[ePlot,], xa=xac[ePlot,], t=ePlot, N=ndof)
        fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear', title='RMSE-Central')

    # plot the iteration statistics and/or error-to-variance ratio
    if ( do_hybrid ):
        fig = plot_iteration_stats(niters)
        fig = plot_error_variance_stats(evratio)
    else:
        if ( nens == 0 ):
            fig = plot_iteration_stats(niters)
        else:
            fig = plot_error_variance_stats(evratio)

    pyplot.show()
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
