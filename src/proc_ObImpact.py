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
from   plot_stats    import *
###############################################################

###############################################################
def main():

    # save figure to disk
    save_fig = False

    # get the name of .dat file to read and the start and end indices
    [_, fname, sOI, eOI] = get_input_arguments()

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

    old_format = False
    if ( ('ens_dJ' in object.keys()) or ('adj_dJ' in object.keys()) ): old_format = True

    if ( old_format ):
        adJa = object['adj_dJa']
        adJb = object['adj_dJb']
        edJa = object['ens_dJa']
        edJb = object['ens_dJb']
        adJ  = object['adj_dJ' ]
        edJ  = object['ens_dJ' ]
    else:
        adJai = object['adj_dJa']
        adJbi = object['adj_dJb']
        edJai = object['ens_dJa']
        edJbi = object['ens_dJb']

        adJa  = np.nansum(adJai,axis=1)
        adJb  = np.nansum(adJbi,axis=1)
        edJa  = np.nansum(edJai,axis=1)
        edJb  = np.nansum(edJbi,axis=1)

        adJ   = adJa + adJb
        edJ   = edJa + edJb

    if ( sOI < 0 ): sOI = 0
    if ( eOI < 0 ): eOI = len(adJ)

    titlestr = '$\delta J$ = $\delta J_a$ + $\delta J_b$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J$'
    fig1 = plot_ObImpact(adJ[sOI:eOI],edJ[sOI:eOI],sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    titlestr = '$\delta J_a$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J_a$'
    fig2 = plot_ObImpact(adJa[sOI:eOI],edJa[sOI:eOI],sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    titlestr = '$\delta J_b$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J_b$'
    fig3 = plot_ObImpact(adJb[sOI:eOI],edJb[sOI:eOI],sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    if ( save_fig ):
        fOrient = 'portrait'
        fig1.savefig(fname_fig + '-dJ.eps', dpi=300,orientation=fOrient,format='eps')
        fig2.savefig(fname_fig + '-dJa.eps',dpi=300,orientation=fOrient,format='eps')
        fig3.savefig(fname_fig + '-dJb.eps',dpi=300,orientation=fOrient,format='eps')
        fig1.savefig(fname_fig + '-dJ.png', dpi=100,orientation=fOrient,format='png')
        fig2.savefig(fname_fig + '-dJa.png',dpi=100,orientation=fOrient,format='png')
        fig3.savefig(fname_fig + '-dJb.png',dpi=100,orientation=fOrient,format='png')
        print 'all done ...'
    else:
        pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
