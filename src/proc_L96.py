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

    # name of output diagnostic file to read
    fname_diag = 'L96_varDA_diag.nc4'
    fname_diag = 'L96_ensDA_diag.nc4'
    fname_diag = 'L96_hybDA_diag.nc4'

    # read dimensions and necessary attributes from the diagnostic file
    try:
        nc = Dataset(fname_diag, mode='r', format='NETCDF4')
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
        print 'Exception occurred during read of ' + fname_diag
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    # read the diag file
    try:
        nc = Dataset(fname_diag, mode='r', format='NETCDF4')

        xt      = np.squeeze(nc.variables['truth'][:,])
        Xb      = np.squeeze(nc.variables['prior'][:,])
        Xa      = np.squeeze(nc.variables['posterior'][:,])
        y       = np.squeeze(nc.variables['obs'][:,])
        if ( do_hybrid ):
            Xb      = np.transpose(Xb, (0,2,1))
            Xa      = np.transpose(Xa, (0,2,1))
            xbm     = np.squeeze(nc.variables['prior_emean'][:,])
            xam     = np.squeeze(nc.variables['posterior_emean'][:,])
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
        print 'Exception occurred during read of ' + fname_diag
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    # compute RMSE in prior, posterior and observations
    xbrmse = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / ndof )
    xarmse = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / ndof )
    xyrmse = np.sqrt( np.sum( (xt -   y)**2          ) / ndof )

    # plot the last state
    fig = plot_L96(obs=y[-1,], ver=xt[-1,], xb=Xb[-1,], xa=Xa[-1,], t=nassim, N=ndof, figNum = 1)

    # plot the RMSE
    fig = plot_rmse(xbrmse, xarmse, yscale='linear')

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
    sys.exit()

###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
