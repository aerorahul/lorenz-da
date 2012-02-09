#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# varDA.py - Functions related to variational data assimilation
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy as np

def ThreeDvar(xb, B, y, R, H, maxiter=1000, alpha=4e-3, cg=True):
# {{{
    '''
    ThreeDvar(xb, B, y, R, H, maxiter=100, alpha=4e-3, cg=True)
    update using 3Dvar algorithm
    '''

    # start with background
    x       = xb.copy()
    niters  = 0
    Jold    = 1e6
    J       = 0
    Binv    = np.linalg.inv(B)

    while ( np.abs(Jold -J) > 1e-5 ):

        if ( niters > maxiter ):
            print 'exceeded maximum iterations allowed'
            break

        Jold = J

        # cost function : 2J(x) = Jb + Jo | Jb = [x-xb]^T B^(-1) [x-xb] | Jo = [y-Hx]^T R^(-1) [y-Hx]
        Jb = 0.5 * np.dot(np.transpose((x - xb)), np.dot(Binv,(x-xb)))
        Jo = 0.5 * np.dot(np.transpose((y - np.dot(H,x))), np.dot(np.linalg.inv(R),(y - np.dot(H,x))))
        J = Jb + Jo

        if ( niters == 0 ): print "initial cost = %10.5f" % J
        print "cost = %10.5f" % J

        # cost function gradient : dJ/dx
        gJ = np.dot(Binv,(x - xb)) - np.dot(np.linalg.inv(R),(y-np.dot(H,x)))

        if ( cg ):
            if ( niters == 0 ):
                x = x - alpha * gJ
                cgJo = gJ
            else:
                beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJo),gJo)
                cgJ = gJ + beta * cgJo
                x = x - alpha * cgJ
                cgJo = cgJ

            gJo = gJ
        else:
            x = x - alpha * gJ

        niters = niters + 1

    print 'final cost = %10.5f after %d iterations' % (J, niters)

    # 3DVAR estimate
    xa = x.copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv( Binv + np.linalg.inv(R) )

    return xa, A, niters
# }}}

def FourDvar(xb, B, y, R, H, maxiter=1000, alpha=4e-3, cg=True):
# {{{
    '''
    FourDvar(xb, B, y, R, H, maxiter=100, alpha=4e-3, cg=True)
    update using 4Dvar algorithm

    currently calls 3Dvar update
    '''

    [xa, A, niters] = ThreeDvar(xb, B, y, R, H, maxiter=maxiter, alpha=alpha, cg=cg)

    return xa, A, niters
# }}}
