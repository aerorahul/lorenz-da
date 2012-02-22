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

def obs_increment_EnKF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    obs_increment_EnKF(obs, obs_err_var, pr_obs_est)
    compute observation increment due to a single observation using traditional EnKF
            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
    '''

    # compute mean and variance of the PRIOR model estimate of the observation
    pr_mean = np.mean(pr_obs_est)
    pr_var  = np.var( pr_obs_est, ddof=1)

    # compute innovation and total variance
    innov  = obs - pr_mean
    totvar = pr_var + obs_err_var

    # update mean and variance of the POSTERIOR model estimate of the observation
    po_var  = 1.0 / ( 1.0 / pr_var + 1.0 / obs_err_var )
    po_mean = po_var * ( pr_mean / pr_var + obs / obs_err_var )

    # generate perturbed observations, adjust so that mean(pert_obs) = observation
    pert_obs = obs + np.sqrt(obs_err_var) * np.random.randn(len(pr_obs_est))
    pert_obs = pert_obs - np.mean(pert_obs) + obs

    # update POSTERIOR model estimate of the observation
    po_obs_est = po_var * ( pr_obs_est / pr_var + pert_obs / obs_err_var )

    # compute observation increment
    obs_inc = po_obs_est - pr_obs_est

    return obs_inc, innov, totvar
# }}}

def obs_increment_EnSRF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    obs_increment_EnSRF(obs, obs_err_var, pr_obs_est)
    compute observation increment due to a single observation using EnSRF
            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
    '''

    # compute mean and variance of the PRIOR model estimate of the observation
    pr_mean = np.mean(pr_obs_est)
    pr_var  = np.var( pr_obs_est, ddof=1)

    # compute innovation and total variance
    innov  = obs - pr_mean
    totvar = pr_var + obs_err_var

    # update mean and variance of the POSTERIOR model estimate of the observation
    po_var  = 1.0 / ( 1.0 / pr_var + 1.0 / obs_err_var )
    po_mean = pr_mean + ( po_var / obs_err_var ) * ( obs - pr_mean )
    beta    = 1.0 / ( 1.0 + np.sqrt(po_var / pr_var) )

    # update POSTERIOR model estimate of the observation
    po_obs_est = po_mean + (1.0 - beta * po_var / obs_err_var) * (pr_obs_est - pr_mean)

    # compute observation increment
    obs_inc = po_obs_est - pr_obs_est

    return obs_inc, innov, totvar
# }}}

def obs_increment_EAKF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    obs_increment_EAKF(obs, obs_err_var, pr_obs_est)
    compute observation increment due to a single observation using EAKF
            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
    '''

    # compute mean and variance of the PRIOR model estimate of the observation
    pr_mean = np.mean(pr_obs_est)
    pr_var  = np.var( pr_obs_est, ddof=1)

    # compute innovation and total variance
    innov  = obs - pr_mean
    totvar = pr_var + obs_err_var

    # update mean and variance of the POSTERIOR model estimate of the observation
    po_var  = 1.0 / ( 1.0 / pr_var + 1.0 / obs_err_var )
    po_mean = po_var * ( pr_mean / pr_var + obs / obs_err_var )

    # update POSTERIOR model estimate of the observation
    po_obs_est = np.sqrt( po_var / pr_var ) * ( pr_obs_est - pr_mean ) + po_mean

    # compute observation increment
    obs_inc = po_obs_est - pr_obs_est

    return obs_inc, innov, totvar
# }}}

def state_increment(obs_inc, pr, pr_obs_est):
# {{{
    '''
    state_increment(obs_inc, pr_obs_est, pr)
    compute state increment due to an observation increment
        obs_inc - observation increment
             pr - prior
     pr_obs_est - prior observation estimate
      state_inc - state increment
    '''

    covariance = np.cov(pr, pr_obs_est, ddof=1)
    state_inc = obs_inc * covariance[0,1] / covariance[1,1]

    return state_inc
# }}}

def compute_cov_factor(dist, cov_cutoff):
# {{{
    '''
    compute_cov_factor(dist, cov_cutoff)
    compute the covariance factor using Gaspari & Cohn polynomial function
          dist - distance between the points
    cov_cutoff - normalized cutoff distance = cutoff_distance / (2 * normalization_factor)
                 normalized cutoff distance = 1 / (2 * 40)
                 localize at 1 point in the 40-variable LE98 model
    '''

    if ( np.abs(dist) >= 2.0*cov_cutoff ):
        cov_factor = 0.0
    elif ( np.abs(dist) <= cov_cutoff ):
        r = np.abs(dist) / cov_cutoff
        cov_factor = ( ( ( -0.25*r + 0.5 )*r + 0.625 )*r - 5.0/3.0 )*(r**2) + 1.0
    else:
        r = np.abs(dist) / cov_cutoff
        cov_factor = ( ( ( ( r/12 - 0.5 )*r +0.625 )*r + 5.0/3.0 )*r -5.0 )*r + 4.0 - 2.0 / (3.0 * r)

    return cov_factor
# }}}
