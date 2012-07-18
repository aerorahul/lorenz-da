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
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import os, sys
import numpy         as     np
import cPickle       as     cPickle
from   matplotlib    import pyplot, cm
from   module_IO     import *
###############################################################

###############################################################
def main():

    # example usage:
    #fill_ObImpact.py -m varyHR_beta=0.75 -f L96_hybDA_ObImpact_ -s 5 -e 5

    [dir_ObImp,fprefix,nH,nR] = get_input_arguments()

    fname_fig = dir_ObImp + os.sep + fprefix + 'varyHR'
    save_fig  = False

    HRMatrix = np.zeros((5,5)) * np.NaN

    for h in range(0,nH):
        for r in range(0,nR):

            fname = dir_ObImp + os.sep + fprefix + 'H' + str(h+1) + 'R' + str(r+1) + '.dat'

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
                mean_adJ = np.mean(adJ)
                mean_edJ = np.mean(edJ)

                HRMatrix[r,h] = mean_edJ - mean_adJ

    fig = pyplot.figure()
    pyplot.hold(True)
    pyplot.imshow(HRMatrix, cmap=cm.get_cmap(name='Oranges',lut=16), interpolation='nearest')
    pyplot.gca().invert_yaxis()
    pyplot.colorbar()
    pyplot.clim(-0.2,1.4)

    locs, labs = pyplot.xticks()
    newlocs = np.arange(0,nH)
    newlabs = np.arange(0,nH)+1
    pyplot.xticks(newlocs, newlabs)
    locs, labs = pyplot.yticks()
    newlocs = np.arange(0,nR)
    newlabs = np.arange(0,nR)+1
    pyplot.yticks(newlocs, newlabs)

    pyplot.xlabel('H', fontsize=14, fontweight='bold')
    pyplot.ylabel('R', fontsize=14, fontweight='bold')
    pyplot.title('$\mathbf{\delta J_e\ -\ \delta J_a}$', fontsize=14, fontweight='bold')

    for h in range(0,nH):
        for r in range(0,nR):

            txtstr = '%5.4f' % HRMatrix[r,h]
            pyplot.text(h-0.225,r-0.0625,txtstr)

    if ( save_fig ):
        fOrient = 'portrait'
        fig.savefig(fname_fig + '.eps', dpi=300,orientation=fOrient,format='eps')
        fig.savefig(fname_fig + '.png', dpi=100,orientation=fOrient,format='png')
        print 'all done ...'
    else:
        pyplot.show()
        sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
