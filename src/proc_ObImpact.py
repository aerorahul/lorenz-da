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
from   plot_stats    import *
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

    adJ  = object['adj_dJ']
    edJ  = object['ens_dJ']
    adJa = object['adj_dJa']
    adJb = object['adj_dJb']
    edJa = object['ens_dJa']
    edJb = object['ens_dJb']

    if sOI < 0: sOI = 0
    if eOI < 0: eOI = len(adJ)

    fig = plot_ObImpact(dJa=adJ[sOI:eOI], dJe=edJ[sOI:eOI], startxIndex=sOI)

    fig = pyplot.figure()
    pyplot.hold(True)
    pyplot.plot(adJa[sOI:eOI],'b-')
    pyplot.plot(edJa[sOI:eOI],'r-')
    pyplot.plot(np.zeros(eOI-sOI+1),'k-')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJa[sOI:eOI]), np.std(adJa[sOI:eOI],ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJa[sOI:eOI]), np.std(edJa[sOI:eOI],ddof=1))
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
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(adJb[sOI:eOI]), np.std(adJb[sOI:eOI],ddof=1))
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(edJb[sOI:eOI]), np.std(edJb[sOI:eOI],ddof=1))
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
