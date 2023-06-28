#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# comp_varDA.py - compare the effects of inflating static cov
#                 on the performance of a variational DA
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
from   argparse   import ArgumentParser,ArgumentDefaultsHelpFormatter
from   netCDF4    import Dataset
from   module_IO  import *
###############################################################

###############################################################
def main():

    # name of starting ensDA output diagnostic file, starting index and measure

    parser = ArgumentParser(description='compare the diag files written by varDA.py',formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--filename',help='name of the diag file to read',required=True)
    parser.add_argument('-m','--measure',help='measure to evaluate performance',required=False,choices=['obs','truth'],default='truth')
    parser.add_argument('-b','--begin_index',help='starting index to read',type=int,required=False,default=101)
    parser.add_argument('-e','--end_index',help='ending index to read',type=int,required=False,default=-1)
    parser.add_argument('-s','--save_figure',help='save figures',action='store_true',required=False)
    args = parser.parse_args()

    fname    = args.filename
    measure  = args.measure
    sOI      = args.begin_index
    eOI      = args.end_index
    save_fig = args.save_figure

    # Inflation factors to compare
    #alpha = [1.0, 2.0, 3.0, 3.1, 3.2, 3.4]
    alpha = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 1.0]
    alpha = [1.0, 2.0, 2.5]

    # some more arguments, currently hard-coded
    save_figures = False         # save plots as eps
    yscale       = 'linear'      # y-axis of RMSE plots (linear/semilog)
    yFix         = 0.18          # fix the y-axis of RMSE plots ( None = automatic )
    fOrient      = 'portrait'    # figure orientation (landscape/portrait)

    if ( not measure ): measure = 'truth'
    if ( sOI == -1 ): sOI = 0

    nf = len(alpha)
    fnames = []
    for i in range(nf): fnames.append( fname + '%3.2f.nc4' % ((alpha[i])) )

    if ( len(fnames) <= 15):
        fcolor = ["#000000", "#C0C0C0", "#808080", "#800000", "#FF0000",\
                  "#800080", "#FF00FF", "#008000", "#00FF00", "#808000",\
                  "#FFFF00", "#000080", "#0000FF", "#008080", "#00FFFF"]
        # black, silver, gray, maroon, red
        # purple, fuchsia, green, lime, olive
        # yellow, navy, blue, teal, aqua
    else:
        fcolor = get_Ndistinct_colors(len(fnames))

    # read general dimensions and necessary attributes from the diagnostic file
    [model, DA, _, gvarDA] = read_diag_info(fnames[0])

    Bc = read_clim_cov(model=model,norm=True)

    if   ( gvarDA.update == 1 ): vstr = '3DVar'
    elif ( gvarDA.update == 2 ): vstr = '4DVar'

    # allocate room for variables
    print('computing RMSE against %s' % measure)
    xbrmse = np.zeros((len(fnames),DA.nassim))
    xarmse = np.zeros((len(fnames),DA.nassim))
    xyrmse = np.zeros((len(fnames),DA.nassim))
    flabel = []
    blabel = []
    mean_prior     = np.zeros(len(fnames))
    mean_posterior = np.zeros(len(fnames))
    std_prior      = np.zeros(len(fnames))
    std_posterior  = np.zeros(len(fnames))
    mean_niters    = np.zeros(len(fnames))
    std_niters     = np.zeros(len(fnames))
    innov          = np.zeros(len(fnames))
    mean_evratio   = np.zeros(len(fnames))
    std_evratio    = np.zeros(len(fnames))

    for fname in fnames:

        print('reading ... %s' % fname)
        f = fnames.index(fname)

        try:
            nc = Dataset(fname, mode='r', format='NETCDF4')
            flabel.append(r'$\alpha = %3.2f$' % alpha[f])
            blabel.append('%3.2f' % alpha[f])
            nc.close()
        except Exception as Instance:
            print('Exception occurred during read of ' + fname)
            print(type(Instance))
            print(Instance.args)
            print(Instance)
            sys.exit(1)

        # read the varDA for the specific diagnostic file
        [_, _, _, varDA] = read_diag_info(fname)

        # read the diagnostic file
        xt, xb, xa, y, H, R, niters = read_diag(fname, 0, end_time=DA.nassim)
        if ( varDA.update == 2 ): y = y[:,:model.Ndof]

        # compute RMSE in prior, posterior and observations
        if ( measure == 'truth' ):
            xbrmse[f,] = np.sqrt( np.sum( (xt - xb)**2, axis = 1) / model.Ndof )
            xarmse[f,] = np.sqrt( np.sum( (xt - xa)**2, axis = 1) / model.Ndof )
        else:
            xbrmse[f,] = np.sqrt( np.sum( (y - xb)**2, axis = 1) / model.Ndof )
            xarmse[f,] = np.sqrt( np.sum( (y - xa)**2, axis = 1) / model.Ndof )
        xyrmse[f,] = np.sqrt( np.sum( (xt -  y)**2 ) / model.Ndof )

        evratio = niters.copy()
        evratio = np.zeros(len(niters))
        for i in range(DA.nassim):
            innov  = np.sum((y[i,:] - np.dot(np.diag(H[i,:]),xb[  i,:]))**2)
            totvar = np.sum(varDA.inflation.infl_fac*np.diag(Bc) + R[i,:])
            evratio[i] = innov / totvar
        mean_evratio[f] = np.mean(evratio[sOI:])
        std_evratio[f]  = np.std( evratio[sOI:],ddof=1)

        # compute mean and std. dev. in the iteration count
        mean_niters[f] = np.mean(niters[sOI+1:])
        std_niters[f]  = np.std( niters[sOI+1:], ddof=1)

    # start plotting

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmse[f,sOI:])
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix is None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(q))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmse[f,sOI:])
        mean_prior[f] = np.mean(q)
        std_prior[f]  = np.std(q,ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Prior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_varDA_RMSE_Prior.eps' % (model.Name),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmse[f,sOI:])
        if   ( yscale == 'linear'  ): pyplot.plot(    q,'-',color=fcolor[f],label=flabel[f],linewidth=1)
        elif ( yscale == 'semilog' ): pyplot.semilogy(q,'-',color=fcolor[f],label=flabel[f],linewidth=1)

    yl = pyplot.get(pyplot.gca(),'ylim')
    xl = pyplot.get(pyplot.gca(),'xlim')
    if ( yFix is None ): ymax = yl[1]
    else:                ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(q))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmse[f,sOI:])
        mean_posterior[f] = np.mean(q)
        std_posterior[f]  = np.std(q,ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q,ddof=1))
        pyplot.text(25,(1-0.05*(f+1))*ymax,str,color=fcolor[f],fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('RMSE - Posterior',fontweight='bold',fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_varDA_RMSE_Posterior.eps' % (model.Name),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.15
    width = 0.35

    bottom = 0.0
    pyplot.bar(index,mean_prior-bottom,width,bottom=bottom,linewidth=0.0,color='0.75',edgecolor='0.75',yerr=std_prior, error_kw=dict(ecolor='black',elinewidth=3,capsize=5))
    pyplot.bar(index+width,mean_posterior-bottom,width,bottom=bottom,linewidth=0.0,color='gray',edgecolor='gray',yerr=std_posterior,error_kw=dict(ecolor='black',elinewidth=3,capsize=5))

    pyplot.xticks(index+width, blabel)

    pyplot.xlabel('Inflation Factor', fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',             fontweight='bold',fontsize=12)
    pyplot.title( 'RMSE',             fontweight='bold',fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_varDA_RMSE.eps' % (model.Name),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.2
    width = 0.6

    pyplot.bar(index,mean_niters,width,linewidth=0.0,color='gray',edgecolor='gray',yerr=std_niters,error_kw=dict(ecolor='black',elinewidth=3,capsize=5))

    pyplot.xticks(index+width/2, blabel)

    pyplot.xlabel('Inflation Factor',  fontweight='bold',fontsize=12)
    pyplot.ylabel('No. of Iterations', fontweight='bold',fontsize=12)
    pyplot.title( 'No. of Iterations', fontweight='bold',fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_varDA_niters.eps' % (model.Name),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------

    #-----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.2
    width = 0.6
    pyplot.bar(index,mean_evratio,width,linewidth=0.0,color='gray',edgecolor='gray',yerr=std_evratio,error_kw=dict(ecolor='black',elinewidth=3,capsize=5))

    pyplot.xticks(index+width/2, blabel)

    pyplot.xlabel('Inflation Factor',       fontweight='bold',fontsize=12)
    pyplot.ylabel('Error - Variance Ratio', fontweight='bold',fontsize=12)
    pyplot.title( 'Error - Variance Ratio', fontweight='bold',fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig('%s_varDA_evratio.eps' % (model.Name),dpi=300,orientation=fOrient,format='eps')
    #-----------------------------------------------------------


    if not save_figures: pyplot.show()
    print('... all done ...')
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
