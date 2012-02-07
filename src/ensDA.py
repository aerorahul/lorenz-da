#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# ensDA.py - Functions related to ensemble data assimilation
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy as np

def EnKF(xbm, Xbp, Y, H, R, loc):
# {{{
    '''
    EnKF(xbm, Xbp, Y, H, R, loc):
    update an ensemble using the EnKF algorithm
    '''

    N = np.shape(Xbp)[0] # number of degrees of freedom in state vector
    M = np.shape(Xbp)[1] # number of ensemble members
    P = np.shape(Y)[0]   # number of observations

    # full vector of model estimated observations and mean
    Ye  = np.dot(H, np.transpose(np.tile(xbm,(M,1))) + Xbp)
    mye = np.mean(Ye,axis=1)

    for ob in range(0,P):

        # vector of model estimated obs with mean removed
        ye = Ye[ob,:] - mye[ob]
        varye = np.var(ye,ddof=1)
        obs_err = R[ob,ob]

        # innovation and innovation variance
        innov     = Y[ob] - mye[ob]
        innov_var = varye + obs_err

        # B H^T ~ X (H X)^T ~ cov(x,ye)
        kcov = np.dot(Xbp,np.transpose(ye)) / (M - 1)

        # localize the gain
        kcov = kcov * np.transpose(loc[ob,:])

        # Kalman gain
        K = kcov / innov_var

        # update the mean
        xam = xbm + K * innov

        # update the ensemble
        beta = 1 / ( 1 + np.sqrt(obs_err / innov_var) )
        Xap = Xbp - np.outer(beta*K,ye)

    return xam, Xap
# }}}

def PerturbedObs(Xb, B, Y, H, R):
# {{{
    '''
    PerturbedObs(Xb, B, Y, H, R):
    update an ensemble by perturbed observations
    '''

    N = np.shape(Xb)[0] # number of degrees of freedom in state vector
    M = np.shape(Xb)[1] # number of ensemble members

    Xa = Xb.copy() # initialize Xa

    for n in range(0,M):
        xb = Xb[:,n].copy()
        Yp = Y + np.diag(np.diag(np.random.randn(N))*np.sqrt(R))
        K  = np.dot(B,np.linalg.inv(B + R))
        Xa[:,n] = Xb[:,n] + np.dot(K, (Yp - Xb[:,n]))

    return Xa
# }}}
def Potter(xbm, Xbp, Y, H, R):
# {{{
    '''
    Potter(xbm, Xbp, Y, H, R):
    update an ensemble using the Potter algorithm
    '''

    N = np.shape(Xbp)[0] # number of degrees of freedom in state vector
    M = np.shape(Xbp)[1] # number of ensemble members
    P = np.shape(Y)[0]   # number of observations

    for ob in range(0,P):
        F     = np.dot(H[ob,:],Xbp)                          # (1 x M) model estimate (y_e)
        alpha = 1/(np.dot(F,np.transpose(F)) + R[ob,ob])     # (scalar) innovation variance (HBH^T + R)
        gamma = 1/(1 + np.sqrt(alpha*R[ob,ob]))              # (scalar) beta for reduced Kalman gain
        K     = alpha*np.dot(Xbp,np.transpose(F))            # Kalman gain (N x 1)
        xam = xbm + np.dot(K, (Y[ob] - np.dot(H[ob,:],xbm))) # update mean
        Xap = Xbp - gamma*np.outer(K,F)                      # update perturbations

    return xam, Xap
# }}}
