#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# fill_ObImpact.py - read observation impact data and process
#                    it for plotting, etc.
###############################################################

###############################################################
from module_IO import *
from matplotlib import pyplot, cm
import numpy as np
import sys
import os
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

    # example usage:
    # fill_ObImpact.py -m varyHR_beta=0.75 -f L96_hybDA_ObImpact_ -s 5 -e 5

    [dir_ObImp, fprefix, nH, nR] = get_input_arguments()

    fname_fig = dir_ObImp + os.sep + fprefix + 'HR'
    save_fig = False

    Matrix = np.zeros((nR, nH)) * np.NaN

    for h in range(0, nH):
        for r in range(0, nR):

            fname = dir_ObImp + os.sep + fprefix + 'H' + \
                str(h + 1) + 'R' + str(r + 1) + '.nc4'

            if ((h == 0) and (r == 0)):
                [model, DA, ensDA, varDA] = read_diag_info(fname)

            [edJai, edJbi, adJai, adJbi] = read_ObImpact_diag(
                fname, 0, end_time=DA.nassim)

            mean_adJ = np.mean(
                np.nansum(
                    adJai,
                    axis=1) +
                np.nansum(
                    adJbi,
                    axis=1))
            mean_edJ = np.mean(
                np.nansum(
                    edJai,
                    axis=1) +
                np.nansum(
                    edJbi,
                    axis=1))

            Matrix[r, h] = mean_edJ - mean_adJ

    fig = pyplot.figure()
    pyplot.hold(True)
    pyplot.imshow(
        Matrix,
        cmap=cm.get_cmap(
            name='PuOr_r',
            lut=64),
        interpolation='nearest')
    pyplot.gca().invert_yaxis()
    pyplot.colorbar()
    pyplot.clim(-1.6, 1.6)

    locs, labs = pyplot.xticks()
    newlocs = np.arange(0, nH)
    newlabs = np.arange(0, nH) + 1
    pyplot.xticks(newlocs, newlabs)
    locs, labs = pyplot.yticks()
    newlocs = np.arange(0, nR)
    newlabs = np.arange(0, nR) + 1
    pyplot.yticks(newlocs, newlabs)

    pyplot.xlabel('H', fontsize=14, fontweight='bold')
    pyplot.ylabel('R', fontsize=14, fontweight='bold')
    pyplot.title(
        '$\\mathbf{\\delta J_e\\ -\\ \\delta J_a}$',
        fontsize=14,
        fontweight='bold')

    for h in range(0, nH):
        for r in range(0, nR):
            pyplot.text(h - 0.225, r - 0.0625, '%5.4f' % Matrix[r, h])

    pyplot.hold(False)

    if (save_fig):
        fOrient = 'portrait'
        fig.savefig(
            fname_fig + '.eps',
            dpi=300,
            orientation=fOrient,
            format='eps')
        fig.savefig(
            fname_fig + '.png',
            dpi=100,
            orientation=fOrient,
            format='png')
        print('all done ...')
    else:
        pyplot.show()
        sys.exit(0)
###############################################################


###############################################################
if __name__ == "__main__":
    main()
###############################################################
