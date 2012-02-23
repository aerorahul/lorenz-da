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

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import sys
import numpy as np
###############################################################

def update_varDA(xb, B, y, R, H, Vupdate=None, minimization=[None, None, None]):
# {{{
    '''
    Update the prior with a variational-based state estimation algorithm to produce a posterior

    xa, A, niters = update_varDA(xb, B, y, R, H, Vupdate=1, minimization = [1000, 4e-3, True])

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
     Vupdate - variational data assimilation algorithm [1 = ThreeDvar]
minimization - minimization parameters [maxiter=1000, alpha=4e-3, cg=True]
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # set defaults:
    maxiter, alpha, cg = minimization
    if ( Vupdate == None ): Vupdate = 1
    if ( maxiter == None ): maxiter = 1000
    if ( alpha   == None ): alpha   = 4e-3
    if ( cg      == None ): cg      = True
    minimization = [maxiter, alpha, cg]

    if ( Vupdate == 1 ):
        xa, A, niters = ThreeDvar(xb, B, y, R, H, minimization)

    elif ( Vupdate == 2 ):
        xa, A, niters = FourDvar(xb, B, y, R, H, minimization)

    else:
        print 'invalid update algorithm ...'
        sys.exit(2)

    return xa, A, niters
# }}}

def ThreeDvar(xb, B, y, R, H, minimization):
# {{{
    '''
    Update the prior with 3Dvar algorithm to produce a posterior

    xa, A, niters = ThreeDvar(xb, B, y, R, H, minimization)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
minimization - minimization parameters [maxiter, alpha, cg]
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # get minimization parameters
    maxiter, alpha, cg = minimization

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
        #print "cost = %10.5f" % J

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

    print 'final cost   = %10.5f after %d iterations' % (J, niters)

    # 3DVAR estimate
    xa = x.copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv( Binv + np.linalg.inv(R) )

    return xa, A, niters
# }}}

def FourDvar(xb, B, y, R, H, minimization):
# {{{
    '''
    Update the prior with 4Dvar algorithm to produce a posterior

    xa, A, niters = FourDvar(xb, B, y, R, H, minimization)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
minimization - minimization parameters [maxiter, alpha, cg]
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function

    currently calls the 3Dvar update
    '''

    xa, A, niters = ThreeDvar(xb, B, y, R, H, minimization)

    return xa, A, niters
# }}}
