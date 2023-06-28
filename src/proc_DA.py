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
from   argparse      import ArgumentParser, ArgumentDefaultsHelpFormatter
from   matplotlib    import pyplot
from   module_IO     import *
from   module_DA     import *
from   plot_stats    import *
###############################################################

###############################################################
def main():

    parser = ArgumentParser(description = 'Process the diag file written by ???DA.py', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--filename',help='name of the diag file to read',type=str,required=True)
    parser.add_argument('-m','--measure',help='measure to evaluate performance',type=str,required=False,choices=['obs','truth'],default='truth')
    parser.add_argument('-b','--begin_index',help='starting index to read',type=int,required=False,default=1)
    parser.add_argument('-t','--plot_index',help='time index to plot',type=int,required=False,default=-1)
    parser.add_argument('-s','--save_figure',help='save figures',action='store_true',required=False)
    args = parser.parse_args()

    fname    = args.filename
    measure  = args.measure
    sStat    = args.begin_index
    ePlot    = args.plot_index
    save_fig = args.save_figure

    if ( not os.path.isfile(fname) ):
        print('%s does not exist' % fname)
        sys.exit(1)

    fname_fig = fname.split('diag.nc4')[0]

    if ( sStat <= -1 ): sStat   = 100

    # get model, DA class data and necessary attributes from the diagnostic file:
    [model, DA, ensDA, varDA] = read_diag_info(fname)

    # print some info so the user knows the script is doing something
    print('no. of assimilation cycles = %d' % DA.nassim)

    # read diagnostics from file
    if ( DA.do_hybrid ):
        xt, Xb, Xa, y, H, R, xbc, xac, niters, evratio = read_diag(fname, 0, end_time=DA.nassim)
    else:
        xt, Xb, Xa, y, H, R, tmpvar                    = read_diag(fname, 0, end_time=DA.nassim)

    if ( DA.maxouter > 1 ):
        Xb = np.squeeze(Xb[:, 0,:])
        Xa = np.squeeze(Xa[:,-1,:])

    if ( hasattr(ensDA,'update') ):
        Xb      = np.transpose(Xb, (0,2,1))
        Xa      = np.transpose(Xa, (0,2,1))
        xbm     = np.mean(Xb, axis=2)
        xam     = np.mean(Xa, axis=2)
    else:
        xbm     = Xb.copy()
        xam     = Xa.copy()

    if ( (DA.do_hybrid) or (hasattr(varDA,'update')) ):
        var_evratio = np.zeros(np.shape(xbm)[0])
        Bs = varDA.inflation.infl_fac * read_clim_cov(model=model,norm=True)
        if ( DA.do_hybrid ): L = localization_operator(model,ensDA.localization)
        for i in range(DA.nassim):
            if ( DA.do_hybrid ): xtmp = xbc
            else:                xtmp = xbm
            innov  = np.nansum((y[i,:model.Ndof] - np.dot(np.diag(H[i,:]),xtmp[i,:]))**2)
            if ( DA.do_hybrid ):
                Bc = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * (np.cov(Xb[i,],ddof=1)*L)
            else:
                Bc = Bs
            totvar = np.sum(np.diag(Bc) + R[i,:])
            var_evratio[i] = innov / totvar

    if ( hasattr(varDA,'update') ):
        if   ( varDA.update == 1 ): vstr = '3DVar'
        elif ( varDA.update == 2 ): vstr = '4DVar'
    if ( hasattr(ensDA,'update') ):
        if   ( ensDA.update == 1 ): estr = 'EnKF'
        elif ( ensDA.update == 2 ): estr = 'EnSRF'
        elif ( ensDA.update == 3 ): estr = 'EAKF'
    if ( DA.do_hybrid ):
        fstr = estr
    else:
        if   ( hasattr(varDA,'update') ): fstr, niters  = vstr, tmpvar
        elif ( hasattr(ensDA,'update') ): fstr, evratio = estr, tmpvar

    # compute RMSE in prior, posterior and observations
    print('computing RMSE against %s' % measure)
    if ( measure == 'truth' ):
        xbrmse = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / model.Ndof )
        xarmse = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / model.Ndof )
    else:
        xbrmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xbm)**2, axis = 1) / model.Ndof )
        xarmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xam)**2, axis = 1) / model.Ndof )
    xyrmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xt )**2 ) / model.Ndof )

    if   ( ePlot == 0 ): pIndex = 0
    elif ( ePlot >  0 ): pIndex = ePlot + 1
    elif ( ePlot <  0 ): pIndex = DA.nassim + ePlot
    if ( (pIndex < 0) or (pIndex >= DA.nassim) ):
        print('ERROR : t = %d does not exist in %s' % (pIndex+1, fname))
        print('        valid options are t = +/- [1 ... %d]' % DA.nassim)
        sys.exit(2)

    # plot the RMSE
    fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear', pretitle=fstr)
    if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-RMSE' % fstr)

    # plot the last state
    if   ( model.Name == 'L63' ):
        fig = plot_trace(obs=y, ver=xt, xb=xbm, xa=xam, N=model.Ndof, pretitle=fstr)
        if ( save_fig ): save_figure(fig, fname = fname_fig + 'attractor')

    elif ( model.Name == 'L96' ):
        fig = plot_L96(obs=y[pIndex,:model.Ndof], ver=xt[pIndex,], xb=Xb[pIndex,], xa=Xa[pIndex,], t=pIndex+1, N=model.Ndof, pretitle=fstr)
        if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-%d' % (fstr,pIndex+1))

    # plot the last state and RMSE for central state
    if ( DA.do_hybrid ):
        if ( measure == 'truth' ):
            xbrmse = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / model.Ndof )
            xarmse = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / model.Ndof )
        else:
            xbrmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xbc)**2, axis = 1) / model.Ndof )
            xarmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xac)**2, axis = 1) / model.Ndof )
        xyrmse = np.sqrt( np.sum( (y[:,:model.Ndof] - xt )**2 ) / model.Ndof )
        fig = plot_rmse(xbrmse=xbrmse, xarmse=xarmse, sStat=sStat, yscale='linear', pretitle=vstr)
        if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-RMSE' % vstr)

        if   ( model.Name == 'L63' ):
            fig = plot_trace(obs=y, ver=xt, xb=xbm, xa=xam, N=model.Ndof, pretitle=vstr)
        elif ( model.Name == 'L96' ):
            fig = plot_L96(obs=y[pIndex,:model.Ndof], ver=xt[pIndex,], xb=xbc[pIndex,], xa=xac[pIndex,], t=pIndex+1, N=model.Ndof, pretitle=vstr)
            if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-%d' % (vstr, pIndex+1))

    # plot the iteration statistics and/or error-to-variance ratio
    if ( hasattr(varDA,'update') ):
        for outer in range(DA.maxouter):
            fig = plot_iteration_stats(np.squeeze(niters[:,outer]),pretitle=vstr)
            if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-niters-%d' % (vstr,outer+1))
        fig = plot_error_variance_stats(var_evratio, sStat=sStat, pretitle=vstr)
        if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-evratio' % vstr)
    if ( hasattr(ensDA,'update') ):
        titlestr = estr + ' - Innovation Variance / Total Variance'
        fig = plot_error_variance_stats(evratio, sStat=sStat, pretitle=estr)
        if ( save_fig ): save_figure(fig, fname = fname_fig + '%s-evratio' % estr)

    if ( not save_fig ): pyplot.show()
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
