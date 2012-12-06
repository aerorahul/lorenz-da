#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# loop_LXX.py - read diagnostics for LXX and make loops
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
import numpy         as     np
from   module_Lorenz import *
from   module_IO     import *
###############################################################

###############################################################
def main():

    # get the name of output diagnostic file to read
    [_, fname, sOI, eOI] = get_input_arguments()
    if ( not os.path.isfile(fname) ):
        print '%s does not exist' % fname
        sys.exit(1)

    fname_fig = fname.split('.nc4')[0]

    # get model, DA class data and necessary attributes from the diagnostic file:
    [model, DA, ensDA, varDA] = read_diag_info(fname)

    # print some info so the user knows the script is doing something
    print 'no. of assimilation cycles = %d' % DA.nassim

    if ( sOI == -1 ): sOI = 0
    if ( eOI == -1 ): eOI = DA.nassim

    # read diagnostics from file
    if ( DA.do_hybrid ):
        xt, Xb, Xa, y, H, R, xbc, xac, niters, evratio = read_diag(fname, 0, end_time=DA.nassim)
    else:
        xt, Xb, Xa, y, H, R, tmpvar                    = read_diag(fname, 0, end_time=DA.nassim)

    if ( hasattr(ensDA,'update') ):
        Xb      = np.transpose(Xb, (0,2,1))
        Xa      = np.transpose(Xa, (0,2,1))
        xbm     = np.mean(Xb, axis=2)
        xam     = np.mean(Xa, axis=2)
    else:
        xbm     = Xb.copy()
        xam     = Xa.copy()

    if ( hasattr(varDA,'update') ):
        if   ( varDA.update == 1 ): vstr = '3DVar'
        elif ( varDA.update == 2 ): vstr = '4DVar'
    if ( hasattr(ensDA,'update') ):
        if   ( ensDA.update == 1 ): estr = 'EnKF'
        elif ( ensDA.update == 2 ): estr = 'EnSRF'
        elif ( ensDA.update == 3 ): estr = 'EAKF'

    if ( DA.do_hybrid ):
        fstr = estr
    else:
        if   ( hasattr(varDA,'update') ): fstr, niters  = vstr, tmpvar
        elif ( hasattr(ensDA,'update') ): fstr, evratio = estr, tmpvar

    # Loop through the states
    for t in range(sOI, eOI):

        fig = plot_L96(obs=y[t,], ver=xt[t,], xb=Xb[t,], xa=Xa[t,], t=t+1, N=model.Ndof, pretitle=fstr)

        fname = fname_fig + '_%04d.png' % (t+1)
        print 'Saving frame', fname
        fig.savefig(fname)

    print '... all done'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
