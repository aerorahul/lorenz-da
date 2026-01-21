#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# mc_ObImpact.py - read observation impact data and use it for
#                  Monte-Carlo processing
###############################################################

###############################################################
from plot_stats import *
from module_IO import *
from matplotlib import pyplot
import pickle as cPickle
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

    # get the name of .dat file to read
    [_, fname, nEns, _] = get_input_arguments()
    if (not os.path.isfile(fname)):
        print('%s does not exist' % fname)
        sys.exit(1)

    try:
        fh = open(fname, 'rb')
        object = cPickle.load(fh)
        fh.close()
    except Exception as Instance:
        print('Exception occured during read of %s' % fname)
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    adJ = object['adj_dJ']
    edJ = object['ens_dJ']
    adJa = object['adj_dJa']
    adJb = object['adj_dJb']
    edJa = object['ens_dJa']
    edJb = object['ens_dJb']

    nSam = len(adJ)
    nSubSam = nSam / nEns
    if (not isinstance(nSubSam, int)):
        print(
            'nSam = %d must be exactly divisible by nEns = %d' %
            (nSam, nEns))
        sys.exit(1)

    print('total no. of samples ................. = %d' % nSam)
    print('no. of ensemble members .............. = %d' % nEns)
    print('no. of samples in each ensemble member = %d' % nSubSam)

    adJ_s = np.zeros((nEns, nSam / nEns)) * np.NaN
    edJ_s = np.zeros((nEns, nSam / nEns)) * np.NaN
    adJa_s = np.zeros((nEns, nSam / nEns)) * np.NaN
    adJb_s = np.zeros((nEns, nSam / nEns)) * np.NaN
    edJa_s = np.zeros((nEns, nSam / nEns)) * np.NaN
    edJb_s = np.zeros((nEns, nSam / nEns)) * np.NaN

    sInd = 0

    for n in range(0, nEns):
        adJ_s[n, :] = adJ[sInd:sInd + nSam / nEns]
        edJ_s[n, :] = edJ[sInd:sInd + nSam / nEns]
        adJa_s[n, :] = adJa[sInd:sInd + nSam / nEns]
        adJb_s[n, :] = adJb[sInd:sInd + nSam / nEns]
        edJa_s[n, :] = edJa[sInd:sInd + nSam / nEns]
        edJb_s[n, :] = edJb[sInd:sInd + nSam / nEns]

        sInd = sInd + nSam / nEns

    adJ_sm = np.mean(adJ_s, axis=1)
    edJ_sm = np.mean(edJ_s, axis=1)
    adJa_sm = np.mean(adJa_s, axis=1)
    adJb_sm = np.mean(adJb_s, axis=1)
    edJa_sm = np.mean(edJa_s, axis=1)
    edJb_sm = np.mean(edJb_s, axis=1)

    titlestr = '$\\delta J$ = $\\delta J_a$ + $\\delta J_b$'
    xlabstr = 'Assimilation Step'
    ylabstr = '$\\delta J$'
    fig1 = plot_ObImpact(
        adJ_sm,
        edJ_sm,
        title=titlestr,
        xlabel=xlabstr,
        ylabel=ylabstr)

    titlestr = '$\\delta J_a$'
    xlabstr = 'Assimilation Step'
    ylabstr = '$\\delta J_a$'
    fig2 = plot_ObImpact(
        adJa_sm,
        edJa_sm,
        title=titlestr,
        xlabel=xlabstr,
        ylabel=ylabstr)

    titlestr = '$\\delta J_b$'
    xlabstr = 'Assimilation Step'
    ylabstr = '$\\delta J_b$'
    fig3 = plot_ObImpact(
        adJb_sm,
        edJb_sm,
        title=titlestr,
        xlabel=xlabstr,
        ylabel=ylabstr)

    pyplot.show()
###############################################################


###############################################################
if __name__ == "__main__":
    main()
###############################################################
