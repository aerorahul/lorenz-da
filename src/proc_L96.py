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
    # state, measure to do verification (truth | observations)
    [measure,fname,sStat,ePlot] = get_input_arguments()
    if ( not os.path.isfile(fname) ):
        print '%s does not exist' % fname
        sys.exit(1)

    if ( sStat <= -1 ): sStat   = 100
    if ( not measure ): measure = 'truth'

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
            if ( (nc.Vupdate == 1) or (nc.Vupdate == 3) ): varDA = 3
            if ( (nc.Vupdate == 2) or (nc.Vupdate == 4) ): varDA = 4

        if 'do_hybrid' in nc.ncattrs():
            do_hybrid = nc.do_hybrid
            if ( (nc.Vupdate == 1) or (nc.Vupdate == 3) ): varDA = 3
            if ( (nc.Vupdate == 2) or (nc.Vupdate == 4) ): varDA = 4
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
    print 'computing RMSE against %s' % measure
    if ( measure == 'truth' ):
        xbrmse = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / ndof )
        xarmse = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / ndof )
    else:
        xbrmse = np.sqrt( np.sum( (y  - xbm)**2, axis = 1) / ndof )
        xarmse = np.sqrt( np.sum( (y  - xam)**2, axis = 1) / ndof )
    xyrmse = np.sqrt( np.sum( (xt - y)**2 ) / ndof )

    if   ( ePlot == 0 ): pIndex = 0
    elif ( ePlot >  0 ): pIndex = ePlot + 1
    elif ( ePlot <  0 ): pIndex = nassim + ePlot
    if ( (pIndex < 0) or (pIndex >= nassim) ):
        print 'ERROR : t = %d does not exist in %s' % (pIndex+1, fname)
        print '        valid options are t = +/- [1 ... %d]' % nassim
        sys.exit(2)

    # plot the last state
    fig = plot_L96(obs=y[pIndex,], ver=xt[pIndex,], xb=Xb[pIndex,], xa=Xa[pIndex,], t=pIndex+1, N=ndof)

    # plot the RMSE
    fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear')

    # plot the last state and RMSE for central state
    if ( do_hybrid ):
        if ( measure == 'truth' ):
            xbrmse = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / ndof )
            xarmse = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / ndof )
        else:
            xbrmse = np.sqrt( np.sum( (y  - xbc)**2, axis = 1) / ndof )
            xarmse = np.sqrt( np.sum( (y  - xac)**2, axis = 1) / ndof )
        xyrmse = np.sqrt( np.sum( (xt - y)**2 ) / ndof )
        fig = plot_L96(obs=y[pIndex,], ver=xt[pIndex,], xb=xbc[pIndex,], xa=xac[pIndex,], t=pIndex+1, N=ndof)
        fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear', title='RMSE-%dDVar'%(varDA))

    # plot the iteration statistics and/or error-to-variance ratio
    if ( do_hybrid ):
        fig = plot_iteration_stats(niters)
        fig = plot_error_variance_stats(evratio, sStat=sStat)
    else:
        if ( nens == 0 ):
            fig = plot_iteration_stats(niters)
        else:
            fig = plot_error_variance_stats(evratio, sStat=sStat)

    pyplot.show()
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
