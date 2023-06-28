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
import os, sys, numpy
from   argparse      import ArgumentParser, ArgumentDefaultsHelpFormatter
from   matplotlib    import pyplot
from   module_Lorenz import plot_L96
from   module_IO     import read_diag_info, read_diag
###############################################################

###############################################################
def main():

    parser = ArgumentParser(description = 'Process the diag file written by ???DA.py', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f','--filename',help='name of the diag file to read',type=str,required=True)
    parser.add_argument('-b','--begin_index',help='starting index to read',type=int,required=False,default=1)
    parser.add_argument('-e','--end_index',help='ending index to read',type=int,required=False,default=-1)
    parser.add_argument('-s','--save_figure',help='save figures',action='store_true',required=False)
    args = parser.parse_args()

    fname    = args.filename
    sOI      = args.begin_index
    eOI      = args.end_index
    save_fig = args.save_figure

    if ( not os.path.isfile(fname) ):
        print('%s does not exist' % fname)
        sys.exit(1)

    fname_fig = fname.split('.nc4')[0]

    # get model, DA class data and necessary attributes from the diagnostic file:
    [model, DA, ensDA, varDA] = read_diag_info(fname)

    # print some info so the user knows the script is doing something
    print('no. of assimilation cycles = %d' % DA.nassim)

    if ( sOI == -1 ): sOI = DA.nassim-1
    if ( eOI == -1 ): eOI = DA.nassim

    # read diagnostics from file
    if ( DA.do_hybrid ):
        xt, Xb, Xa, y, H, R, xbc, xac, niters, evratio = read_diag(fname, 0, end_time=DA.nassim)
    else:
        xt, Xb, Xa, y, H, R, tmpvar                    = read_diag(fname, 0, end_time=DA.nassim)

    if ( hasattr(ensDA,'update') ):
        Xb      = numpy.transpose(Xb, (0,2,1))
        Xa      = numpy.transpose(Xa, (0,2,1))
        xbm     = numpy.mean(Xb, axis=2)
        xam     = numpy.mean(Xa, axis=2)
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

        fig = plot_L96(obs=y[t,], ver=xt[t,], xb=Xb[t,], xa=Xa[t,], t=t, N=model.Ndof, pretitle=fstr, figNum=1)

        fname = fname_fig + '_%05d.png' % (t)
        if ( save_fig ):
            print('Saving frame %d as %s'  % (t, fname))
            fig.savefig(fname)
        else:
            print('Showing frame %d as %s' % (t, fname))
            pyplot.pause(0.1)

    if ( not save_fig ): pyplot.show()
    print('... all done')
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
