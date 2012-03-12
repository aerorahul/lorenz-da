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
import numpy         as     np
import cPickle       as     cPickle
from   matplotlib    import pyplot
from   plot_stats    import *
###############################################################

###############################################################
def main():

    model = 'L96'

    fh = open('../data/' + model + '/ensDA_N=40/inf=1.21/L96_ensDA_ObImpact.dat','rb')
    object = cPickle.load(fh)
    fh.close()

    adJ  = object['adj_dJ']
    edJ  = object['ens_dJ']
    adJa = object['adj_dJa']
    adJb = object['adj_dJb']
    edJa = object['ens_dJa']
    edJb = object['ens_dJb']

    sOI = 500
    eOI = 1000

    fig = plot_ObImpact(dJa=adJ[sOI:eOI], dJe=edJ[sOI:eOI], startxIndex=sOI)
    #fig.savefig('ObImpact200.eps',dpi=300,orientation='landscape',format='eps')

    fig = pyplot.figure()
    pyplot.hold(True)
    pyplot.plot(adJa[sOI:eOI],'b-')
    pyplot.plot(edJa[sOI:eOI],'r-')
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJa), np.std(adJa,ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJa), np.std(edJa,ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    yoff = yl[0] + 0.4
    pyplot.text(5,yoff,stra,fontsize=10)
    yoff = yl[0] + 0.2
    pyplot.text(5,yoff,stre,fontsize=10)
    pyplot.title('a-component')
    pyplot.hold(False)

    fig = pyplot.figure()
    pyplot.hold(True)
    pyplot.plot(adJb[sOI:eOI],'b-')
    pyplot.plot(edJb[sOI:eOI],'r-')
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJb), np.std(adJb,ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJb), np.std(edJb,ddof=1))
    yl = pyplot.get(pyplot.gca(),'ylim')
    yoff = yl[0] + 0.4
    pyplot.text(5,yoff,stra,fontsize=10)
    yoff = yl[0] + 0.2
    pyplot.text(5,yoff,stre,fontsize=10)
    pyplot.title('b-component')
    pyplot.hold(False)

    pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
