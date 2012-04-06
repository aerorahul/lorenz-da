#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# comp_hybDA.py - compare the effects of different weights on
#                 static B on the performance of a hybrid DA
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
import numpy      as     np
from   matplotlib import pyplot
from   netCDF4    import Dataset
from   module_IO  import *
###############################################################

###############################################################
def main():

    # name of starting hybrid output diagnostic file, starting index and no. of files
    [measure, fname, sOI, nf] = get_input_arguments()

    if ( not measure ): measure = 'truth'

    fnames = []
    for i in range(0,nf): fnames.append(fname.replace('e0','e%d'%i))

    fcolor = ['black', 'gray', 'blue', 'red', 'green', 'cyan', 'magenta']
    if ( len(fnames) > 7 ): fcolor = get_Ndistinct_colors(len(fnames))
    save_figures = False

    # read dimensions and necessary attributes from the diagnostic file
    try:
        nc = Dataset(fnames[0], mode='r', format='NETCDF4')
        ndof   = len(nc.dimensions['ndof'])
        nassim = len(nc.dimensions['ntime'])
        nobs   = len(nc.dimensions['nobs'])
        nens   = len(nc.dimensions['ncopy'])

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

    # allocate room for variables
    print 'computing RMSE against %s' % measure
    xbrmseE = np.zeros((len(fnames),nassim))
    xarmseE = np.zeros((len(fnames),nassim))
    xbrmseC = np.zeros((len(fnames),nassim))
    xarmseC = np.zeros((len(fnames),nassim))
    xyrmse  = np.zeros((len(fnames),nassim))
    flabel  = []

    for fname in fnames:

        print 'reading ... %s' % fname
        f = fnames.index(fname)

        try:
            nc = Dataset(fname, mode='r', format='NETCDF4')
            flabel.append(r'$\beta_e$ = %3.2f' % nc.hybrid_wght)
            nc.close()
        except Exception as Instance:
            print 'Exception occurred during read of ' + fname
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

        # read the diagnostic file
        xt, Xb, Xa, y, _, _, xbc, xac, _, _ = read_diag(fname, 0, end_time=nassim)

        # compute ensemble mean
        xbm = np.squeeze(np.mean(Xb, axis=1))
        xam = np.squeeze(np.mean(Xa, axis=1))

        # compute RMSE in prior, posterior and observations
        if ( measure == 'truth' ):
            xbrmseE[f,] = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / ndof )
            xarmseE[f,] = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / ndof )
            xbrmseC[f,] = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / ndof )
            xarmseC[f,] = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / ndof )
        else:
            xbrmseE[f,] = np.sqrt( np.sum( (y - xbm)**2, axis = 1) / ndof )
            xarmseE[f,] = np.sqrt( np.sum( (y - xam)**2, axis = 1) / ndof )
            xbrmseC[f,] = np.sqrt( np.sum( (y - xbc)**2, axis = 1) / ndof )
            xarmseC[f,] = np.sqrt( np.sum( (y - xac)**2, axis = 1) / ndof )
        xyrmse[f,]  = np.sqrt( np.sum( (xt - y)**2          ) / ndof )

    # start plotting
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseE[f,sOI:])
        pyplot.plot(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    pyplot.ylim(0.0, yl[1])
    pyplot.xlim(0.0, len(np.squeeze(xbrmseE[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseE[f,sOI:])
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*yl[1],str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Ensemble Prior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('RMSE_HybridEnKF_Prior.eps',dpi=300,orientation='landscape',format='eps')

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseC[f,sOI:])
        pyplot.plot(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    pyplot.ylim(0.0, yl[1])
    pyplot.xlim(0.0, len(np.squeeze(xbrmseC[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseC[f,sOI:])
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*yl[1],str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Central Prior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('RMSE_Hybrid3DVar_Prior.eps',dpi=300,orientation='landscape',format='eps')

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseE[f,sOI:])
        pyplot.plot(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    pyplot.ylim(0.0, yl[1])
    pyplot.xlim(0.0, len(np.squeeze(xarmseE[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseE[f,sOI:])
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*yl[1],str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Ensemble Posterior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('RMSE_HybridEnKF_Posterior.eps',dpi=300,orientation='landscape',format='eps')

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseC[f,sOI:])
        pyplot.plot(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    pyplot.ylim(0.0, yl[1])
    pyplot.xlim(0.0, len(np.squeeze(xarmseC[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseC[f,sOI:])
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*yl[1],str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Central Posterior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('RMSE_Hybrid3DVar_Posterior.eps',dpi=300,orientation='landscape',format='eps')

    if not save_figures: pyplot.show()
    sys.exit(0)
###############################################################

###############################################################
def get_Ndistinct_colors(num_colors):
    from colorsys import hls_to_rgb
    colors=[]
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue        = i/360.0
        lightness  = (50 + np.random.rand() * 10)/100.0
        saturation = (90 + np.random.rand() * 10)/100.0
        colors.append(hls_to_rgb(hue, lightness, saturation))
    return colors
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
