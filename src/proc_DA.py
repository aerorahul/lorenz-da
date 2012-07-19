#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# proc_DA.py - process the diagnostics written by ???DA.py
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

    # get model, DA class data and necessary attributes from the diagnostic file:
    [model, DA, ensDA, varDA] = read_diag_info(fname)

    # print some info so the user knows the script is doing something
    print 'no. of assimilation cycles = %d' % DA.nassim

    # read diagnostics from file
    if ( DA.do_hybrid ):
        xt, Xb, Xa, y, H, R, xbc, xac, niters, evratio = read_diag(fname, 0, end_time=DA.nassim)
    else:
        xt, Xb, Xa, y, H, R, tmpvar                    = read_diag(fname, 0, end_time=DA.nassim)

    if ( ( not DA.do_hybrid ) and ( hasattr(ensDA,'update') ) ): evratio = tmpvar
    if ( ( not DA.do_hybrid ) and ( hasattr(varDA,'update') ) ): niters  = tmpvar

    if ( hasattr(ensDA,'update') ):
        Xb      = np.transpose(Xb, (0,2,1))
        Xa      = np.transpose(Xa, (0,2,1))
        xbm     = np.mean(Xb, axis=2)
        xam     = np.mean(Xa, axis=2)
    else:
        xbm     = Xb.copy()
        xam     = Xa.copy()

    # compute RMSE in prior, posterior and observations
    print 'computing RMSE against %s' % measure
    if ( measure == 'truth' ):
        xbrmse = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / model.Ndof )
        xarmse = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / model.Ndof )
    else:
        xbrmse = np.sqrt( np.sum( (y  - xbm)**2, axis = 1) / model.Ndof )
        xarmse = np.sqrt( np.sum( (y  - xam)**2, axis = 1) / model.Ndof )
    xyrmse = np.sqrt( np.sum( (xt - y)**2 ) / model.Ndof )

    if   ( ePlot == 0 ): pIndex = 0
    elif ( ePlot >  0 ): pIndex = ePlot + 1
    elif ( ePlot <  0 ): pIndex = DA.nassim + ePlot
    if ( (pIndex < 0) or (pIndex >= DA.nassim) ):
        print 'ERROR : t = %d does not exist in %s' % (pIndex+1, fname)
        print '        valid options are t = +/- [1 ... %d]' % DA.nassim
        sys.exit(2)

    # plot the RMSE
    fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear')

    # plot the last state
    if   ( model.Name == 'L63' ):
        fig = plot_trace(obs=y, ver=xt, xb=xbm, xa=xam, N=model.Ndof)
    elif ( model.Name == 'L96' ):
        fig = plot_L96(obs=y[pIndex,], ver=xt[pIndex,], xb=Xb[pIndex,], xa=Xa[pIndex,], t=pIndex+1, N=model.Ndof)

    # plot the last state and RMSE for central state
    if ( DA.do_hybrid ):
        if ( measure == 'truth' ):
            xbrmse = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / model.Ndof )
            xarmse = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / model.Ndof )
        else:
            xbrmse = np.sqrt( np.sum( (y  - xbc)**2, axis = 1) / model.Ndof )
            xarmse = np.sqrt( np.sum( (y  - xac)**2, axis = 1) / model.Ndof )
        xyrmse = np.sqrt( np.sum( (xt - y)**2 ) / model.Ndof )
        if ( varDA.update in [1,3] ): titlestr = 'RMSE-3DVar'
        if ( varDA.update in [2,4] ): titlestr = 'RMSE-4DVar'
        fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear', title=titlestr)
        if   ( model.Name == 'L63' ):
            fig = plot_trace(obs=y, ver=xt, xb=xbm, xa=xam, N=model.Ndof)
        elif ( model.Name == 'L96' ):
            fig = plot_L96(obs=y[pIndex,], ver=xt[pIndex,], xb=xbc[pIndex,], xa=xac[pIndex,], t=pIndex+1, N=model.Ndof)

    # plot the iteration statistics and/or error-to-variance ratio
    if ( hasattr(varDA,'update') ): fig = plot_iteration_stats(niters)
    if ( hasattr(ensDA,'update') ): fig = plot_error_variance_stats(evratio, sStat=sStat)

    pyplot.show()
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
