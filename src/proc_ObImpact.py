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
from   matplotlib    import pyplot
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
def main():

    # save figure to disk
    save_fig = False

    # get the name of ob. impact file to read and the start and end indices
    [_, fname, sOI, eOI] = get_input_arguments()

    [model, DA, ensDA, varDA] = read_diag_info(fname)

    if ( sOI < 0 ): sOI = 0
    if ( eOI < 0 ): eOI = DA.nassim

    [edJai, edJbi, adJai, adJbi] = read_ObImpact_diag(fname,sOI,end_time=eOI)

    edJa = np.nansum(edJai,axis=1)
    edJb = np.nansum(edJbi,axis=1)
    adJa = np.nansum(adJai,axis=1)
    adJb = np.nansum(adJbi,axis=1)

    edJ = edJa + edJb
    adJ = adJa + adJb

    edJi = edJai + edJbi
    adJi = adJai + adJbi

    adJm = np.nansum(adJi,axis=0) / (eOI - sOI)
    edJm = np.nansum(edJi,axis=0) / (eOI - sOI)

    titlestr = '$\delta J$ = $\delta J_a$ + $\delta J_b$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J$'
    fig1 = plot_ObImpact(adJ,edJ,sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    titlestr = '$\delta J_a$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J_a$'
    fig2 = plot_ObImpact(adJa,edJa,sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    titlestr = '$\delta J_b$'
    xlabstr  = 'Assimilation Step'
    ylabstr  = '$\delta J_b$'
    fig3 = plot_ObImpact(adJb,edJb,sOI=sOI,eOI=eOI,title=titlestr,xlabel=xlabstr,ylabel=ylabstr)

    titlestr = 'mean ( $\delta J_a$ )'
    exec('fig4 = plot_ObImpact_%s(adJm, N=%d, t=titlestr)' % (model.Name, model.Ndof))
    titlestr = 'mean ( $\delta J_e$ )'
    exec('fig5 = plot_ObImpact_%s(edJm, N=%d, t=titlestr)' % (model.Name, model.Ndof))

    if ( save_fig ):

        fOrient = 'portrait'
        fname_fig = fname.split('.nc4')[0]

        fig1.savefig(fname_fig + '-dJ.eps'  , dpi=300,orientation=fOrient,format='eps')
        fig2.savefig(fname_fig + '-dJa.eps' , dpi=300,orientation=fOrient,format='eps')
        fig3.savefig(fname_fig + '-dJb.eps' , dpi=300,orientation=fOrient,format='eps')

        fig1.savefig(fname_fig + '-dJ.png',   dpi=100,orientation=fOrient,format='png')
        fig2.savefig(fname_fig + '-dJa.png',  dpi=100,orientation=fOrient,format='png')
        fig3.savefig(fname_fig + '-dJb.png',  dpi=100,orientation=fOrient,format='png')

        fig4.savefig(fname_fig + '-ens_dJm.eps', dpi=300,orientation=fOrient,format='eps')
        fig5.savefig(fname_fig + '-adj_dJm.eps', dpi=300,orientation=fOrient,format='eps')

        fig4.savefig(fname_fig + '-ens_dJm.png', dpi=100,orientation=fOrient,format='png')
        fig5.savefig(fname_fig + '-adj_dJm.png', dpi=100,orientation=fOrient,format='png')

    if ( save_fig ):
        print 'all done ...'
    else:
        pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
