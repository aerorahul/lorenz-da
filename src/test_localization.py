#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# test_localization.py - test effect of localization on static
#                        and flow-dependent covariance.
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
from   matplotlib    import pyplot, cm
from   module_DA     import *
from   module_IO     import *
###############################################################

###############################################################
def main():

    fdiag = '/home/rmahajan/svn-work/lorenz1963/data/varyHR_beta=0.75/diag/L96_hybDA_diag_H1R1.nc4'

    [model, DA, ensDA, varDA]       = read_diag_info(fdiag)
    [_, Xb, _, _, _, _, _, _, _, _] = read_diag(fdiag, 0)
    Xbp = np.transpose(Xb - np.mean(Xb,axis=0))
    Xb  = np.transpose(Xb)

    ensDA.localization.cov_cutoff = 0.0125 * 5

    Bs = read_clim_cov(model)
    Be = np.cov(Xb,ddof=1)

    L  = np.zeros((model.Ndof,model.Ndof))
    L2 = np.zeros((model.Ndof,model.Ndof))
    for i in range(0,model.Ndof):
        for j in range(0,model.Ndof):
            dist = np.float( np.abs( i - j ) ) / model.Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            cov_factor = compute_cov_factor(dist, ensDA.localization.cov_cutoff)
            L[i,j]  = cov_factor
            L2[i,j] = np.sqrt(cov_factor)
            print 'i = %2d, j = %2d, d = %5.3f, c = %10.8f, c = %10.8f' % (i, j, dist, L[i,j], L2[i,j])

    XbpL2 = np.zeros((model.Ndof,model.Ndof*ensDA.Nens))
    for i in range(0,ensDA.Nens):
        start = i*model.Ndof
        end   = i*model.Ndof + model.Ndof
        XbpL2[:,start:end] = np.dot(np.diag(Xbp[:,i]),L2) / np.sqrt(ensDA.Nens-1)

    Be_L2 = np.dot(XbpL2,np.transpose(XbpL2))

    fig1 = plot_cov(Bs,         "$\mathbf{B_s}$")
    fig2 = plot_cov(Be,         "$\mathbf{B_e}$")
    fig3 = plot_cov(Bs*L,       "$\mathbf{B_s}\circ\mathbf{L}$")
    fig4 = plot_cov(Be*L,       "$\mathbf{B_e}\circ\mathbf{L}$")
    fig5 = plot_cov(Be_L2,      "$[\mathbf{X^'}\mathbf{L}][\mathbf{X^'}\mathbf{L}]^T$")
    fig6 = plot_cov(Bs-Bs*L,    "$\mathbf{B_s}\ -\ \mathbf{B_s}\circ\mathbf{L}$")
    fig7 = plot_cov(Be-Be*L,    "$\mathbf{B_e}\ -\ \mathbf{B_e}\circ\mathbf{L}$")
    fig8 = plot_cov(Be-Be_L2,   "$\mathbf{B_e}\ -\ [\mathbf{X^'}\mathbf{L}][\mathbf{X^'}\mathbf{L}]^T$")
    fig9 = plot_cov(Be*L-Be_L2, "$\mathbf{B_e}\circ\mathbf{L}\ - [\mathbf{X^'}\mathbf{L}][\mathbf{X^'}\mathbf{L}]^T$")

    pyplot.show()
    sys.exit(0)

def plot_cov(B,titlestr):

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    cmax = np.round(np.max(np.abs(B)),2)
    pyplot.imshow(B, cmap=cm.get_cmap(name='PuOr_r', lut=128), interpolation='nearest')
    pyplot.gca().invert_yaxis()
    pyplot.colorbar()
    pyplot.clim(-cmax,cmax)

    newlocs = np.arange(4,np.shape(B)[0],5)
    newlabs = newlocs + 1
    pyplot.xticks(newlocs, newlabs)
    pyplot.yticks(newlocs, newlabs)

    pyplot.xlabel('N',     fontsize=12, fontweight='bold')
    pyplot.ylabel('N',     fontsize=12, fontweight='bold')
    pyplot.title(titlestr, fontsize=14, fontweight='bold')

    return fig
###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
