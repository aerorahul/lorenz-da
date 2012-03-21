#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# proc_ObImpact.py - read observation impact data and process
#                    it for plotting, etc.
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
import cPickle       as     cPickle
from   matplotlib    import pyplot
from   module_IO     import *
###############################################################

###############################################################
def main():

    # get the name of .dat file to read and the start and end indices
    [fname, sOI, eOI] = get_input_arguments()

    if ( not os.path.isfile(fname) ):
        print '%s does not exist' % fname
        sys.exit(1)
    else:
        try:
            fh = open(fname,'rb')
            object = cPickle.load(fh)
            fh.close()
        except Exception as Instance:
            print 'Exception occured during read of %s' % fname
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

    fname_fig = fname.split('.dat')[0]

    adJ  = object['adj_dJ']
    edJ  = object['ens_dJ']
    adJa = object['adj_dJa']
    adJb = object['adj_dJb']
    edJa = object['ens_dJa']
    edJb = object['ens_dJb']

    if sOI < 0: sOI = 0
    if eOI < 0: eOI = len(adJ)

    index = np.arange(eOI-sOI)
    width = 0.45
    color_adj = 'c'
    color_ens = 'm'

    fig1 = pyplot.figure()
    pyplot.hold(True)
    r1 = pyplot.bar(index,       adJ[sOI:eOI], width, color=color_adj, edgecolor=color_adj)
    r2 = pyplot.bar(index+width, edJ[sOI:eOI], width, color=color_ens, edgecolor=color_ens)
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJ[sOI:eOI]), np.std(adJ[sOI:eOI],ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJ[sOI:eOI]), np.std(edJ[sOI:eOI],ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    dyl = yl[1] - yl[0]
    yoff = yl[0] + 0.1 * dyl
    pyplot.text(5,yoff,stra,fontsize=10,color=color_adj)
    yoff = yl[0] + 0.2 * dyl
    pyplot.text(5,yoff,stre,fontsize=10,color=color_ens)
    pyplot.title(r'$\delta J$ = $\delta J_a$ + $\delta J_b$', fontsize=14)
    pyplot.xlabel('Assimilation Step', fontsize=12)
    pyplot.ylabel(r'$\delta J$', fontsize=12)
    pyplot.hold(False)

    fig2 = pyplot.figure()
    pyplot.hold(True)
    r1 = pyplot.bar(index,       adJa[sOI:eOI], width, color=color_adj, edgecolor=color_adj)
    r2 = pyplot.bar(index+width, edJa[sOI:eOI], width, color=color_ens, edgecolor=color_ens)
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJa[sOI:eOI]), np.std(adJa[sOI:eOI],ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJa[sOI:eOI]), np.std(edJa[sOI:eOI],ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    dyl = yl[1] - yl[0]
    yoff = yl[0] + 0.1 * dyl
    pyplot.text(5,yoff,stra,fontsize=10,color=color_adj)
    yoff = yl[0] + 0.2 * dyl
    pyplot.text(5,yoff,stre,fontsize=10,color=color_ens)
    pyplot.title(r'$\delta J_a$', fontsize=14)
    pyplot.xlabel('Assimilation Step', fontsize=12)
    pyplot.ylabel(r'$\delta J_a$', fontsize=12)
    pyplot.hold(False)

    fig3 = pyplot.figure()
    pyplot.hold(True)
    r1 = pyplot.bar(index,       adJa[sOI:eOI], width, color=color_adj, edgecolor=color_adj)
    r2 = pyplot.bar(index+width, edJa[sOI:eOI], width, color=color_ens, edgecolor=color_ens)
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJb[sOI:eOI]), np.std(adJb[sOI:eOI],ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJb[sOI:eOI]), np.std(edJb[sOI:eOI],ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    dyl = yl[1] - yl[0]
    yoff = yl[0] + 0.1 * dyl
    pyplot.text(5,yoff,stra,fontsize=10,color=color_adj)
    yoff = yl[0] + 0.2 * dyl
    pyplot.text(5,yoff,stre,fontsize=10,color=color_ens)
    pyplot.title(r'$\delta J_b$', fontsize=14)
    pyplot.xlabel('Assimilation Step', fontsize=12)
    pyplot.ylabel(r'$\delta J_b$', fontsize=12)
    pyplot.hold(False)

    fig1.savefig(fname_fig + '-dJ.eps', dpi=300,orientation='landscape',format='eps')
    fig2.savefig(fname_fig + '-dJa.eps',dpi=300,orientation='landscape',format='eps')
    fig3.savefig(fname_fig + '-dJb.eps',dpi=300,orientation='landscape',format='eps')

    pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
