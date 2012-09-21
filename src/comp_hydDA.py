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

    # some more arguments, currently hard-coded
    save_figures = False         # save plots as eps
    yscale       = 'linear'      # y-axis of RMSE plots (linear/semilog)
    yFix         = None          # fix the y-axis of RMSE plots ( None = automatic )
    fOrient      = 'portrait'    # figure orientation (landscape/portrait)

    if ( not measure ): measure = 'truth'

    if ( sOI == -1 ): sOI = 0

    fnames = []
    for i in range(0,nf): fnames.append(fname.replace('e0','e%d'%i))

    fcolor = ['black', 'gray', 'blue', 'red', 'green', 'cyan', 'magenta']
    if ( len(fnames) > 7 ): fcolor = get_Ndistinct_colors(len(fnames))

    # read dimensions and necessary attributes from the diagnostic file
    [model, DA, ensDA, varDA] = read_diag_info(fnames[0])

    if   ( ensDA.update == 1 ): estr = 'EnKF'
    elif ( ensDA.update == 2 ): estr = 'EnSRF'
    elif ( ensDA.update == 3 ): estr = 'EAKF'
    if   ( varDA.update == 1 ): vstr = '3D'
    elif ( varDA.update == 2 ): vstr = '4D'

    # allocate room for variables
    print 'computing RMSE against %s' % measure
    xbrmseE = np.zeros((len(fnames),DA.nassim))
    xarmseE = np.zeros((len(fnames),DA.nassim))
    xbrmseC = np.zeros((len(fnames),DA.nassim))
    xarmseC = np.zeros((len(fnames),DA.nassim))
    xyrmse  = np.zeros((len(fnames),DA.nassim))
    flabel  = []
    blabel  = []
    mean_prior_C     = np.zeros(len(fnames))
    mean_prior_E     = np.zeros(len(fnames))
    mean_posterior_C = np.zeros(len(fnames))
    mean_posterior_E = np.zeros(len(fnames))
    std_prior_C      = np.zeros(len(fnames))
    std_prior_E      = np.zeros(len(fnames))
    std_posterior_C  = np.zeros(len(fnames))
    std_posterior_E  = np.zeros(len(fnames))

    for fname in fnames:

        print 'reading ... %s' % fname
        f = fnames.index(fname)

        try:
            nc = Dataset(fname, mode='r', format='NETCDF4')
            flabel.append(r'$\beta_e$ = %3.2f' % nc.hybrid_wght)
            blabel.append('%3.2f' % nc.hybrid_wght)
            nc.close()
        except Exception as Instance:
            print 'Exception occurred during read of ' + fname
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

        # read the diagnostic file
        xt, Xb, Xa, y, _, _, xbc, xac, _, _ = read_diag(fname, 0, end_time=DA.nassim)

        # compute ensemble mean
        xbm = np.squeeze(np.mean(Xb, axis=1))
        xam = np.squeeze(np.mean(Xa, axis=1))

        # compute RMSE in prior, posterior and observations
        if ( measure == 'truth' ):
            xbrmseE[f,] = np.sqrt( np.sum( (xt - xbm)**2, axis = 1) / model.Ndof )
            xarmseE[f,] = np.sqrt( np.sum( (xt - xam)**2, axis = 1) / model.Ndof )
            xbrmseC[f,] = np.sqrt( np.sum( (xt - xbc)**2, axis = 1) / model.Ndof )
            xarmseC[f,] = np.sqrt( np.sum( (xt - xac)**2, axis = 1) / model.Ndof )
        else:
            xbrmseE[f,] = np.sqrt( np.sum( (y - xbm)**2, axis = 1) / model.Ndof )
            xarmseE[f,] = np.sqrt( np.sum( (y - xam)**2, axis = 1) / model.Ndof )
            xbrmseC[f,] = np.sqrt( np.sum( (y - xbc)**2, axis = 1) / model.Ndof )
            xarmseC[f,] = np.sqrt( np.sum( (y - xac)**2, axis = 1) / model.Ndof )
        xyrmse[f,] = np.sqrt( np.sum( (xt - y)**2 ) / model.Ndof )

    # start plotting

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseE[f,sOI:])
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix == None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(np.squeeze(xbrmseE[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseE[f,sOI:])
        mean_prior_E[f] = np.mean(q)
        std_prior_E[f]  = np.std(q,ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %s Prior' % estr,fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%s_Prior.eps' % (model.Name, vstr, estr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseC[f,sOI:])
        mean_prior_C[f] = np.mean(q)
        std_prior_C[f]  = np.std(q,ddof=1)
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix == None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(np.squeeze(xbrmseC[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmseC[f,sOI:])
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %sVar Prior' % (vstr), fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%sVar_Prior.eps' % (model.Name, vstr, vstr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseE[f,sOI:])
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix == None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(np.squeeze(xarmseE[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseE[f,sOI:])
        mean_posterior_E[f] = np.mean(q)
        std_posterior_E[f]  = np.std(q,ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %s Posterior' % estr,fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%s_Posterior.eps' % (model.Name, vstr, estr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseC[f,sOI:])
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix == None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(np.squeeze(xarmseC[f,sOI:])))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmseC[f,sOI:])
        mean_posterior_C[f] = np.mean(q)
        std_posterior_C[f]  = np.std(q,ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %sVar Posterior' % (vstr),fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%sVar_Posterior.eps' % (model.Name, vstr, vstr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.15
    width = 0.35

    pyplot.bar(index,mean_prior_E,width,linewidth=0.0,color='0.75',edgecolor='0.75',yerr=std_prior_E, error_kw=dict(ecolor='black',elinewidth=3,capsize=5))
    pyplot.bar(index+width,mean_posterior_E,width,linewidth=0.0,color='gray',edgecolor='gray',yerr=std_posterior_E,error_kw=dict(ecolor='black',elinewidth=3,capsize=5))

    pyplot.xticks(index+width, blabel)

    pyplot.xlabel(r'$\beta_e$',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',      fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %s' % estr,fontweight='bold',fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%s.eps' % (model.Name, vstr, estr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.15
    width = 0.35

    pyplot.bar(index,mean_prior_C,width,linewidth=0.0,color='0.75',edgecolor='0.75',yerr=std_prior_C, error_kw=dict(ecolor='black',elinewidth=3,capsize=5))
    pyplot.bar(index+width,mean_posterior_C,width,linewidth=0.0,color='gray',edgecolor='gray',yerr=std_posterior_C,error_kw=dict(ecolor='black',elinewidth=3,capsize=5))

    pyplot.xticks(index+width, blabel)

    pyplot.xlabel(r'$\beta_e$',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',      fontweight='bold',fontsize=12)
    pyplot.title('RMSE - %sVar' % (vstr),fontweight='bold',fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_%shybDA_RMSE_%sVar.eps' % (model.Name, vstr, vstr),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    if not save_figures: pyplot.show()
    print '... all done ...'
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
