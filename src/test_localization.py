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
#                        Currently implements the following:
#                        Lorenc 2003 Schur operator method
#                        Buehner 2005 method
#                        Liu et al. 2009 method (buggy)
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
from   matplotlib    import pyplot
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import plot_cov
###############################################################

###############################################################
def main():

    fdiag = '/home/rmahajan/svn-work/lorenz1963/test/localization/L96_hybDA_diag_H1R1.nc4'

    [model, DA, ensDA, varDA]       = read_diag_info(fdiag)
    if ( DA.do_hybrid ):[_, Xb, _, _, _, _, _, _, _, _] = read_diag(fdiag, 0)
    else:               [_, Xb, _, _, _, _, _]          = read_diag(fdiag, 0)
    Xbp = np.transpose(Xb - np.mean(Xb,axis=0))
    Xb  = np.transpose(Xb)

    Bs = read_clim_cov(model=model)
    Be = np.cov(Xb,ddof=1)

    L  = np.ones((model.Ndof,model.Ndof))
    L2 = np.ones((model.Ndof,model.Ndof))
    for i in range(0,model.Ndof):
        for j in range(0,model.Ndof):
            dist = np.float( np.abs( i - j ) ) / model.Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            cov_factor = compute_cov_factor(dist, ensDA.localization)
            L[i,j]  = cov_factor
            L2[i,j] = np.sqrt(cov_factor)
            print 'i = %2d, j = %2d, d = %5.3f, c = %10.8f, c = %10.8f' % (i, j, dist, L[i,j], L2[i,j])

    XbpLb = np.zeros((model.Ndof,model.Ndof*ensDA.Nens))
    XbpLl = np.zeros((model.Ndof,model.Ndof*ensDA.Nens))
    for i in range(0,ensDA.Nens):
        start = i*model.Ndof
        end   = i*model.Ndof + model.Ndof
        XbpLb[:,start:end] = np.dot(np.diag(Xbp[:,i]),L2) / np.sqrt(ensDA.Nens-1)
        XbpLl[:,start:end] = L2 * np.repeat(np.transpose(np.matrix(Xbp[:,i])),model.Ndof,axis=1) / np.sqrt(ensDA.Nens-1)
    Be_Lb = np.dot(XbpLb,np.transpose(XbpLb))
    Be_Ll = np.dot(XbpLl,np.transpose(XbpLl))

    fig1  = plot_cov(Bs,          title="Static : $\mathbf{B}_s$")
    fig2  = plot_cov(Be,          title="Ensemble : $\mathbf{B}_e$")
    fig3  = plot_cov(Bs*L,        title="Static Schur : $\mathbf{B}_s \circ\ \mathbf{L}$")
    fig4  = plot_cov(Be*L,        title="Ensemble Schur : $\mathbf{B}_e \circ\ \mathbf{L}$")
    fig5  = plot_cov(Be_Lb,       title="Ensemble Buehner : $[\mathbf{X}^'_b \mathbf{L}] [\mathbf{X}^'_b \mathbf{L}]^{T}$")
    fig6  = plot_cov(Be_Ll,       title="Ensemble Liu")
    fig7  = plot_cov(Bs-Bs*L,     title="Difference Static - Static Schur")
    fig8  = plot_cov(Be-Be*L,     title="Difference Ensemble - Ensemble Schur")
    fig9  = plot_cov(Be-Be_Lb,    title="Difference Ensemble - Ensemble Buehner")
    fig10 = plot_cov(Be-Be_Ll,    title="Difference Ensemble - Ensemble Liu")
    fig11 = plot_cov(Be*L-Be_Lb,  title="Difference Ensemble Schur - Ensemble Buehner")
    fig12 = plot_cov(Be*L-Be_Ll,  title="Difference Ensemble Schur - Ensemble Liu")
    fig13 = plot_cov(Be_Lb-Be_Ll, title="Difference Ensemble Buehner - Ensemble Liu")

    pyplot.show()
    sys.exit(0)

###############################################################

###############################################################
if __name__ == "__main__": main()
###############################################################
