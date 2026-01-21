#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# comp_ensDA.py - compare the effects of varying ensemble size
#                 on the performance of a ensemble DA
###############################################################

###############################################################
from module_IO import *
from netCDF4 import Dataset
from matplotlib import pyplot
import numpy as np
import sys
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################
###############################################################

###############################################################


def main():

    # name of starting ensDA output diagnostic file, starting index and measure
    [measure, fname, sOI, _] = get_input_arguments()

    # Ensemble sizes to compare
    Ne = [5, 10, 20, 30, 40, 50]

    # some more arguments, currently hard-coded
    save_figures = False         # save plots as eps
    yscale = 'linear'      # y-axis of RMSE plots (linear/semilog)
    yFix = None          # fix the y-axis of RMSE plots ( None = automatic )
    fOrient = 'portrait'    # figure orientation (landscape/portrait)

    if (not measure):
        measure = 'truth'
    if (sOI == -1):
        sOI = 0

    nf = len(Ne)
    fnames = []
    for i in range(0, nf):
        fnames.append(fname + '%d.nc4' % Ne[i])
    for i in range(0, nf):
        print(fnames[i])

    if (len(fnames) <= 15):
        fcolor = ["#000000", "#C0C0C0", "#808080", "#800000", "#FF0000",
                  "#800080", "#FF00FF", "#008000", "#00FF00", "#808000",
                  "#FFFF00", "#000080", "#0000FF", "#008080", "#00FFFF"]
        # black, silver, gray, maroon, red
        # purple, fuchsia, green, lime, olive
        # yellow, navy, blue, teal, aqua
    else:
        fcolor = get_Ndistinct_colors(len(fnames))

    # read dimensions and necessary attributes from the diagnostic file
    [model, DA, ensDA, _] = read_diag_info(fnames[0])

    if (ensDA.update == 1):
        estr = 'EnKF'
    elif (ensDA.update == 2):
        estr = 'EnSRF'
    elif (ensDA.update == 3):
        estr = 'EAKF'

    # allocate room for variables
    print('computing RMSE against %s' % measure)
    xbrmse = np.zeros((len(fnames), DA.nassim))
    xarmse = np.zeros((len(fnames), DA.nassim))
    xyrmse = np.zeros((len(fnames), DA.nassim))
    flabel = []
    blabel = []
    mean_prior = np.zeros(len(fnames))
    mean_posterior = np.zeros(len(fnames))
    std_prior = np.zeros(len(fnames))
    std_posterior = np.zeros(len(fnames))
    mean_evratio = np.zeros(len(fnames))
    std_evratio = np.zeros(len(fnames))

    for fname in fnames:

        print('reading ... %s' % fname)
        f = fnames.index(fname)

        try:
            nc = Dataset(fname, mode='r', format='NETCDF4')
            flabel.append('Ne = %d' % len(nc.dimensions['ncopy']))
            blabel.append('%d' % len(nc.dimensions['ncopy']))
            nc.close()
        except Exception as Instance:
            print('Exception occurred during read of ' + fname)
            print(type(Instance))
            print(Instance.args)
            print(Instance)
            sys.exit(1)

        # read the diagnostic file
        xt, Xb, Xa, y, _, _, evratio = read_diag(fname, 0, end_time=DA.nassim)

        # compute ensemble mean
        xbm = np.squeeze(np.mean(Xb, axis=1))
        xam = np.squeeze(np.mean(Xa, axis=1))

        # compute RMSE in prior, posterior and observations
        if (measure == 'truth'):
            xbrmse[f,] = np.sqrt(np.sum((xt - xbm)**2, axis=1) / model.Ndof)
            xarmse[f,] = np.sqrt(np.sum((xt - xam)**2, axis=1) / model.Ndof)
        else:
            xbrmse[f,] = np.sqrt(np.sum((y - xbm)**2, axis=1) / model.Ndof)
            xarmse[f,] = np.sqrt(np.sum((y - xam)**2, axis=1) / model.Ndof)
        xyrmse[f,] = np.sqrt(np.sum((xt - y)**2) / model.Ndof)

        # compute mean and std. dev. in the error-variance ratio
        mean_evratio[f] = np.mean(evratio[sOI + 1:])
        std_evratio[f] = np.std(evratio[sOI + 1:], ddof=1)

    # start plotting

    # -----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmse[f, sOI:])
        if (yscale == 'linear'):
            pyplot.plot(q, '-', color=fcolor[f], label=flabel[f], linewidth=1)
        elif (yscale == 'semilog'):
            pyplot.semilogy(
                q,
                '-',
                color=fcolor[f],
                label=flabel[f],
                linewidth=1)

    yl = pyplot.get(pyplot.gca(), 'ylim')
    xl = pyplot.get(pyplot.gca(), 'xlim')
    if (yFix is None):
        ymax = yl[1]
    else:
        ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(q))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xbrmse[f, sOI:])
        mean_prior[f] = np.mean(q)
        std_prior[f] = np.std(q, ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q, ddof=1))
        pyplot.text(25, (1 - 0.05 * (f + 1)) * ymax,
                    str, color=fcolor[f], fontsize=10)

    pyplot.xlabel('Assimilation Cycle', fontweight='bold', fontsize=12)
    pyplot.ylabel('RMSE', fontweight='bold', fontsize=12)
    pyplot.title('RMSE - Prior', fontweight='bold', fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig(
            '%s_ensDA_RMSE_Prior.eps' %
            (model.Name),
            dpi=300,
            orientation=fOrient,
            format='eps')
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmse[f, sOI:])
        if (yscale == 'linear'):
            pyplot.plot(q, '-', color=fcolor[f], label=flabel[f], linewidth=1)
        elif (yscale == 'semilog'):
            pyplot.semilogy(
                q,
                '-',
                color=fcolor[f],
                label=flabel[f],
                linewidth=1)

    yl = pyplot.get(pyplot.gca(), 'ylim')
    xl = pyplot.get(pyplot.gca(), 'xlim')
    if (yFix is None):
        ymax = yl[1]
    else:
        ymax = yFix
    pyplot.ylim(0.0, ymax)
    pyplot.xlim(0.0, len(q))

    for fname in fnames:
        f = fnames.index(fname)
        q = np.squeeze(xarmse[f, sOI:])
        mean_posterior[f] = np.mean(q)
        std_posterior[f] = np.std(q, ddof=1)
        str = 'mean rmse : %5.4f +/- %5.4f' % (np.mean(q), np.std(q, ddof=1))
        pyplot.text(25, (1 - 0.05 * (f + 1)) * ymax,
                    str, color=fcolor[f], fontsize=10)

    pyplot.xlabel('Assimilation Cycle', fontweight='bold', fontsize=12)
    pyplot.ylabel('RMSE', fontweight='bold', fontsize=12)
    pyplot.title('RMSE - Posterior', fontweight='bold', fontsize=14)
    pyplot.legend(loc=1)
    pyplot.hold(False)
    if save_figures:
        fig.savefig(
            '%s_ensDA_RMSE_Posterior.eps' %
            (model.Name),
            dpi=300,
            orientation=fOrient,
            format='eps')
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.15
    width = 0.35

    pyplot.bar(
        index,
        mean_prior,
        width,
        linewidth=0.0,
        color='0.75',
        edgecolor='0.75',
        yerr=std_prior,
        error_kw=dict(
            ecolor='black',
            elinewidth=3,
            capsize=5))
    pyplot.bar(
        index + width,
        mean_posterior,
        width,
        linewidth=0.0,
        color='gray',
        edgecolor='gray',
        yerr=std_posterior,
        error_kw=dict(
            ecolor='black',
            elinewidth=3,
            capsize=5))

    pyplot.xticks(index + width, blabel)

    pyplot.xlabel('Ensemble Size', fontweight='bold', fontsize=12)
    pyplot.ylabel('RMSE', fontweight='bold', fontsize=12)
    pyplot.title('RMSE', fontweight='bold', fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig(
            '%s_ensDA_RMSE.eps' %
            (model.Name),
            dpi=300,
            orientation=fOrient,
            format='eps')
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)

    index = np.arange(nf) + 0.2
    width = 0.6

    pyplot.bar(
        index,
        mean_evratio,
        width,
        linewidth=0.0,
        color='gray',
        edgecolor='gray',
        yerr=std_evratio,
        error_kw=dict(
            ecolor='black',
            elinewidth=3,
            capsize=5))

    pyplot.xticks(index + width / 2, blabel)

    pyplot.xlabel('Ensemble Size', fontweight='bold', fontsize=12)
    pyplot.ylabel('Error - Variance Ratio', fontweight='bold', fontsize=12)
    pyplot.title('Error - Variance Ratio', fontweight='bold', fontsize=14)
    pyplot.hold(False)
    if save_figures:
        fig.savefig(
            '%s_ensDA_evratio.eps' %
            (model.Name),
            dpi=300,
            orientation=fOrient,
            format='eps')
    # -----------------------------------------------------------

    if not save_figures:
        pyplot.show()
    print('... all done ...')
    sys.exit(0)
###############################################################

###############################################################


def get_Ndistinct_colors(num_colors):
    from colorsys import hls_to_rgb
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(hls_to_rgb(hue, lightness, saturation))
    return colors
###############################################################


###############################################################
if __name__ == "__main__":
    main()
###############################################################
