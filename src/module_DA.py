#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# module_DA.py - Functions related to data assimilation
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import os
import sys
import numpy as np
from scipy import integrate
from module_Lorenz import *

###############################################################

###############################################################
def check_ensDA(Eupdate, inflation, localization):
# {{{
    '''
    Check for valid ensemble DA algorithms and methods

      Eupdate - ensemble data assimilation algorithms
    inflation - inflation
 localization - localization
    '''

    print '==========================================='

    fail = False

    if   ( Eupdate == 0 ):
        print 'Running "No Assimilation"'
    elif ( Eupdate == 1 ):
        print 'Assimilate observations using the EnKF'
    elif ( Eupdate == 2 ):
        print 'Assimilate observations using the EnSRF'
    elif ( Eupdate == 3 ):
        print 'Assimilate observations using the EAKF'
    else:
        print 'Invalid assimilation algorithm'
        print 'Eupdate must be one of : 0 | 1 | 2 | 3'
        print 'No Assimilation | EnKF | EnSRF | EAKF'
        fail = True

    if   ( inflation[0] == 1 ):
        print 'Inflating the Prior using multiplicative inflation with a factor of %f' % inflation[1]
    elif ( inflation[0] == 2 ):
        print 'Inflating the Prior by adding white-noise with zero-mean and %f spread' % inflation[1]
    elif ( inflation[0] == 3 ):
        print 'Inflating the Posterior by covariance relaxation method with weight %f to the prior' % inflation[1]
    elif ( inflation[0] == 4 ):
        print 'Inflating the Posterior by spread restoration method with a factor of %f' % inflation[1]
    else:
        print 'Invalid inflation method'
        print 'inflation[0] must be one of : 1 | 2 | 3 | 4'
        print 'Multiplicative | Additive | Covariance Relaxation | Spread Restoration'
        fail = True

    if   ( localization[0] == True ):
        print 'Localizing using Gaspari-Cohn with a covariance cutoff of %f' % localization[1]
    else:
        print 'No localization'

    print '==========================================='

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_ensDA(Xb, y, R, H, Eupdate=None, inflation=[None, None], localization=[None,
        None]):
# {{{
    '''
    Update the prior with an ensemble-based state estimation algorithm to produce a posterior

    Xa, A, error_variance_ratio = update_ensDA(Xb, B, y, R, H, Eupdate=3, inflation=[1, 1.02], localization=[True, 1.0])

          Xb - prior ensemble
           y - observations
           R - observation error covariance
           H - forward operator
     Eupdate - ensemble-based data assimilation algorithm [3 = EAKF]
   inflation - inflation settings [method, factor = 1, 1.02]
localization - localization settings [localize, cutoff = True, 1.0]
          Xa - posterior ensemble
     evratio - ratio of innovation variance to total variance
    '''

    # set defaults:
    if ( Eupdate         == None ): Eupdate         = 2
    if ( inflation[0]    == None ): inflation[0]    = 1
    if ( inflation[1]    == None ): inflation[1]    = 1.02
    if ( localization[0] == None ): localization[0] = True
    if ( localization[1] == None ): localization[1] = 1.0

    Nobs = np.shape(y)[0]
    Ndof = np.shape(Xb)[0]
    Nens = np.shape(Xb)[1]

    innov  = np.zeros(Nobs) * np.NaN
    totvar = np.zeros(Nobs) * np.NaN

    # prior inflation
    if ( (inflation[0] == 1) or (inflation[0] == 2) ):

        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        if   ( inflation[0] == 1 ): # multiplicative inflation
            Xbp = inflation[1] * Xbp

        elif ( inflation[0] == 2 ): # additive white model error (mean:zero, spread:inflation[1])
            Xbp = Xbp + inflation[1] * np.random.randn(Ndof,Nens)

        Xb = np.transpose(np.transpose(Xbp) + xbm)

    temp_ens = Xb.copy()

    for ob in range(0, Nobs):

        ye = np.dot(H[ob,:],temp_ens)

        if   ( Eupdate == 0 ): # no assimilation
            obs_inc, innov[ob], totvar[ob] = np.zeros(Ndof), 0.0, 0.0

        elif ( Eupdate == 1 ): # update using the EnKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnKF(y[ob], R[ob,ob], ye)

        elif ( Eupdate == 2 ): # update using the EnSRF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnSRF(y[ob], R[ob,ob], ye)

        elif ( Eupdate == 3 ): # update using the EAKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EAKF(y[ob], R[ob,ob], ye)

        else:
            print 'invalid update algorithm ...'
            sys.exit(2)

        for i in range(0,Ndof):
            state_inc = state_increment(obs_inc, temp_ens[i,:], ye)

            # localization
            if ( localization[0] ):
                dist = np.abs( ob - i ) / Ndof
                if ( dist > 0.5 ): dist = 1.0 - dist
                cov_factor = compute_cov_factor(dist, localization[1])
            else:
                cov_factor = 1.0

            temp_ens[i,:] = temp_ens[i,:] + state_inc * cov_factor

    Xa = temp_ens.copy()

    # compute analysis mean and perturbations
    xam = np.mean(Xa,axis=1)
    Xap = np.transpose(np.transpose(Xa) - xam)

    # posterior inflation
    if   ( inflation[0] == 3 ): # covariance relaxation (Zhang & Snyder)
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)
        Xap = Xbp * inflation[1] + Xap * (1.0 - inflation[1])

    elif ( inflation[0] == 4 ): # posterior spread restoration (Whitaker & Hammill)
        xbs = np.std(Xb,axis=1,ddof=1)
        xas = np.std(Xa,axis=1,ddof=1)
        for i in np.arange(0,Ndof):
            Xap[i,:] =  np.sqrt((inflation[1] * (xbs[i] - xas[dof])/xas[i]) + 1.0) * Xap[i,:]

    # add inflated perturbations back to analysis mean
    Xa = np.transpose(np.transpose(Xap) + xam)

    # check for filter divergence
    error_variance_ratio = np.sum(innov**2) / np.sum(totvar)
    if not ( 0.5 < error_variance_ratio < 2.0 ):
        print 'FILTER DIVERGENCE : ERROR / TOTAL VARIANCE = %f' % (error_variance_ratio)
        #break

    return Xa, error_variance_ratio
# }}}
###############################################################

###############################################################
def obs_increment_EnKF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    compute observation increment due to a single observation using traditional EnKF

    obs_inc, innov, totvar = obs_increment_EnKF(obs, obs_err_var, pr_obs_est)

            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
          innov - innovation
         totvar - total variance
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
###############################################################

###############################################################
def obs_increment_EnSRF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    compute observation increment due to a single observation using EnSRF

    obs_inc, innov, totvar = obs_increment_EnSRF(obs, obs_err_var, pr_obs_est)

            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
          innov - innovation
         totvar - total variance
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
###############################################################

###############################################################
def obs_increment_EAKF(obs, obs_err_var, pr_obs_est):
# {{{
    '''
    compute observation increment due to a single observation using EAKF

    obs_inc, innov, totvar = obs_increment_EAKF(obs, obs_err_var, pr_obs_est)

            obs - observation
    obs_err_var - observation error variance
     pr_obs_est - prior observation estimate
        obs_inc - observation increment
          innov - innovation
         totvar - total variance
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
###############################################################

###############################################################
def state_increment(obs_inc, pr, pr_obs_est):
# {{{
    '''
    compute state increment by regressing an observation increment on the state

    state_inc = state_increment(obs_inc, pr, pr_obs_est)

        obs_inc - observation increment
             pr - prior
     pr_obs_est - prior observation estimate
      state_inc - state increment
    '''

    covariance = np.cov(pr, pr_obs_est, ddof=1)
    state_inc = obs_inc * covariance[0,1] / covariance[1,1]

    return state_inc
# }}}
###############################################################

###############################################################
def compute_cov_factor(dist, cov_cutoff):
# {{{
    '''
    compute the covariance factor using Gaspari & Cohn polynomial function

    cov_factor = compute_cov_factor(dist, cov_cutoff)

          dist - distance between "points"
    cov_cutoff - normalized cutoff distance = cutoff_distance / (2 * normalization_factor)
                 Eg. normalized cutoff distance = 1 / (2 * 40)
                     localize at 1 point in the 40-variable LE98 model
    cov_factor - covariance factor
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
###############################################################

###############################################################
def check_varDA(varDA):
# {{{
    '''
    Check for valid variational DA algorithms

    check_varDA(varDA)

    varDA - variational data assimilation class
    '''

    print '==========================================='

    fail = False

    if   ( varDA.update == 0 ):
        print 'Running "No Assimilation"'
    elif ( varDA.update == 1 ):
        print 'Assimilate observations using 3DVar'
    elif ( varDA.update == 2 ):
        print 'Assimilate observations using 4DVar'
    elif ( varDA.update == 3 ):
        print 'Assimilate observations using incremental 3DVar'
    elif ( varDA.update == 4 ):
        print 'Assimilate observations using incremental 4DVar'
    else:
        print 'Invalid assimilation algorithm'
        print 'varDA.update must be one of : 0 | 1 | 2 | 3 | 4'
        print 'No Assimilation | 3DVar | 4DVar | incremental 3DVar | incremental 4DVar'
        fail = True

    print '==========================================='

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_varDA(xb, B, y, R, H, varDA, model=None):
# {{{
    '''
    Update the prior with a variational-based state estimation algorithm to produce a posterior

    xa, A, niters = update_varDA(xb, B, y, R, H, varDA, model=None)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    if   ( varDA.update == 0 ):
        xa, A, niters = xb, B, 0

    elif ( varDA.update == 1 ):
        xa, A, niters = ThreeDvar(xb, B, y, R, H, varDA.minimization)

    elif ( varDA.update == 2 ):
        xa, A, niters = FourDvar(xb, B, y, R, H, varDA.minimization, model, varDA.fdvar)

    elif ( varDA.update == 3 ):
        xa, A, niters = ThreeDvar_inc(xb, B, y, R, H, varDA.minimization)

    elif ( varDA.update == 4 ):
        xa, A, niters = FourDvar_inc(xb, B, y, R, H, varDA.minimization, model, varDA.fdvar)

    else:
        print 'invalid update algorithm ...'
        sys.exit(2)

    return xa, A, niters
# }}}
###############################################################

###############################################################
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
minimization - minimization class
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # start with background
    x       = xb.copy()
    niters  = 0
    Jold    = 1e6
    J       = 0
    Binv    = np.linalg.inv(B)
    Rinv    = np.linalg.inv(R)

    while ( np.abs(Jold - J) > minimization.tol ):

        if ( niters > minimization.maxiter ):
            print 'exceeded maximum iterations allowed'
            break

        Jold = J

        # cost function: J(x) = Jb + Jy
        # Jb = 0.5 * [x-xb]^T B^{-1} [x-xb]
        # Jy = 0.5 * [Hx-y]^T R^{-1} [Hx-y]
        # cost function gradient: gJ = gJb + gJy
        # gJb = B^{-1} [x-xb]
        # gJy = H^T R^{-1} [Hx-y]

        dx  = x - xb
        Jb  = 0.5 * np.dot(np.transpose(dx),np.dot(Binv,dx))
        gJb = np.dot(Binv,dx)

        dy  = np.dot(H,x) - y
        Jy  = 0.5 * np.dot(np.transpose(dy),np.dot(Rinv,dy))
        gJy = np.dot(np.transpose(H),np.dot(Rinv,dy))

        J  =  Jb +  Jy
        gJ = gJb + gJy

        if ( niters == 0 ): print "initial cost = %10.5f" % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        if ( minimization.cg ):
            if ( niters == 0 ):
                x = x - minimization.alpha * gJ
                cgJold = gJ
            else:
                beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJold),gJold)
                cgJ = gJ + beta * cgJold
                x = x - minimization.alpha * cgJ
                cgJold = cgJ

            gJold = gJ
        else:
            x = x - minimization.alpha * gJ

        niters = niters + 1

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # 3DVAR estimate
    xa = x.copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + Rinv)

    return xa, A, niters
# }}}
###############################################################

###############################################################
def ThreeDvar_inc(xb, B, y, R, H, minimization):
# {{{
    '''
    Update the prior with incremental 3Dvar algorithm to produce a posterior

    xa, A, niters = ThreeDvar_inc(xb, B, y, R, H, minimization)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
minimization - minimization class
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # start with zero analysis increment
    dx      = np.zeros(np.shape(xb))
    d       = y - np.dot(H,xb)
    niters  = 0
    Jold    = 1e6
    J       = 0
    Binv    = np.linalg.inv(B)
    Rinv    = np.linalg.inv(R)

    while ( np.abs(Jold - J) > minimization.tol ):

        if ( niters > minimization.maxiter ):
            print 'exceeded maximum iterations allowed, cost = %10.5f' % J
            break

        Jold = J

        # cost function: J(dx) = Jb + Jy
        # Jb = 0.5 * [dx]^T B^{-1} [dx]
        # Jy = 0.5 * [Hdx-d]^T R^{-1} [Hdx-d]
        # cost function gradient: gJ = gJb + gJy
        # gJb = B^{-1} [dx]
        # gJy = H^T R^{-1} [Hdx-d]

        Jb = 0.5 * np.dot(np.transpose(dx),np.dot(Binv,dx))
        Jy = 0.5 * np.dot(np.transpose(np.dot(H,dx)-d),np.dot(Rinv,np.dot(H,dx)-d))
        J = Jb + Jy

        gJb = np.dot(Binv,dx)
        gJy = np.dot(np.transpose(H),np.dot(Rinv,np.dot(H,dx)-d))
        gJ  = gJb + gJy

        if ( niters == 0 ): print "initial cost = %10.5f" % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        if ( minimization.cg ):
            if ( niters == 0 ):
                dx = dx - minimization.alpha * gJ
                cgJold = gJ
            else:
                beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJold),gJold)
                cgJ = gJ + beta * cgJold
                dx = dx - minimization.alpha * cgJ
                cgJold = cgJ

            gJold = gJ
        else:
            dx = dx - minimization.alpha * gJ

        niters = niters + 1

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # 3DVAR estimate
    xa = xb + dx

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + Rinv)

    return xa, A, niters
# }}}
###############################################################

###############################################################
def FourDvar(xb, B, y, R, H, minimization, model, fdvar):
# {{{
    '''
    Update the prior with 4Dvar algorithm to produce a posterior

    xa, A, niters = FourDvar(xb, B, y, R, H, minimization, model, fdvar)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
minimization - minimization class
       model - model class
       fdvar - 4DVar class
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # start with background
    x       = xb.copy()
    niters  = 0
    alpha_r = minimization.alpha
    Jold    = 1e6
    J       = 0
    Binv    = np.linalg.inv(B)
    Rinv    = np.linalg.inv(R)

    while ( np.abs(Jold - J) > minimization.tol ):

        if ( niters > minimization.maxiter ):
            print 'exceeded maximum iterations allowed'
            break

        Jold  = J

        # advance the background through the assimilation window with the full non-linear model
        exec('xnl = integrate.odeint(%s, x, fdvar.twind, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))

        # cost function : J(x) = Jb + Jo
        # Jb = 0.5 * [x-xb]^T B^(-1) [x-xb]
        # Jy = 0.5 * \sum [Hx_i-y_i]^T R^(-1) [Hx_i -y_i]
        # cost function gradient: gJ = gJb + gJy
        # gJb = B^(-1) [x-xb]
        # gJy = \sum M^T H^T R^(-1) [Hx_i -y_i]

        dx  = x - xb
        Jb  = 0.5 * np.dot(np.transpose(dx),np.dot(Binv,dx))
        gJb = np.dot(Binv,dx)

        Jy  = 0.0
        gJy = np.zeros(np.shape(x))
        for j in range(0,fdvar.nobstimes):

            i = fdvar.nobstimes - j - 1
            dy = np.dot(H,xnl[fdvar.twind_obsIndex[i],:]) - y[i,:]
            tint = fdvar.twind[fdvar.twind_obsIndex[i-1]:fdvar.twind_obsIndex[i]+1]

            Jy = Jy + 0.5 * np.dot(np.transpose(dy),np.dot(Rinv,dy))

            gJy = gJy + np.dot(np.transpose(H),np.dot(Rinv,dy))
            if ( len(tint) != 0 ):
                exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (%f,np.flipud(xnl),fdvar.twind, True))' % (model.Name, model.Par[0]+model.Par[1]))
                gJy = sxi[-1,:].copy()

        J  =  Jb +  Jy
        gJ = gJb + gJy

        if ( niters == 0 ): print "initial cost = %10.5f" % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        if ( minimization.cg ):
            if ( niters == 0 ):
                x = x - alpha_r * gJ
                cgJold = gJ
            else:
                beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJold),gJold)
                cgJ = gJ + beta * cgJold
                x = x - alpha_r * cgJ
                cgJold = cgJ

            gJold = gJ
        else:
            x = x - alpha_r * gJ

        niters = niters + 1

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # advance to the analysis time to get the 4DVAR estimate
    exec('xs = integrate.odeint(%s, x, fdvar.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
    xa = xs[-1,:].copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + Rinv)

    return xa, A, niters
# }}}
###############################################################

###############################################################
def FourDvar_inc(xb, B, y, R, H, minimization, model, fdvar):
# {{{
    '''
    Update the prior with incremental 4Dvar algorithm to produce a posterior

    xa, A, niters = FourDvar_inc(xb, B, y, R, H, minimization, model, fdvar)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
minimization - minimization class
       model - model class
       fdvar - 4DVar class
          xa - posterior
           A - analysis error covariance from Hessian
      niters - number of iterations required for minimizing the cost function
    '''

    # start with background
    x       = xb.copy()
    dxo     = np.zeros(np.shape(xb))
    d       = np.zeros(np.shape(y))
    niters  = 0
    alpha_r = minimization.alpha
    Jold    = 1e6
    J       = 1e5
    Binv    = np.linalg.inv(B)
    Rinv    = np.linalg.inv(R)

    for outer in range(0,fdvar.maxouter):

        # advance the background through the assimilation window with full non-linear model
        exec('xnl = integrate.odeint(%s, x, fdvar.twind, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))

        # get observational increment at all observation times
        for i in range(0,fdvar.nobstimes):
            d[i,:] = np.dot(H,xnl[fdvar.twind_obsIndex[i],:]) - y[i,:]

        while ( np.abs(Jold - J) > minimization.tol ):

            if ( niters > minimization.maxiter ):
                print 'exceeded maximum iterations allowed'
                break

            Jold = J

            # advance the increment through the assimilation window with TL model
            exec('dxtl = integrate.odeint(%s_tlm, dxo, fdvar.twind, (%f,xnl,fdvar.twind,False))' % (model.Name, model.Par[0]+model.Par[1]))

            # cost function : J(dx_o) = Jb + Jy
            # Jb = 0.5 * [dx_o]^T B^(-1) [dx_o]
            # Jy = 0.5 * \sum [Hdx_i-d_i]^T R^(-1) [Hdx_i-d_i]
            # cost function gradient: gJ = gJb + gJy
            # gJb = B^(-1) [dx_o]
            # gJy = \sum M^T H^T R^(-1) [Hdx_i-d_i]

            Jb  = 0.5 * np.dot(np.transpose(dxo),np.dot(Binv,dxo))
            gJb = np.dot(Binv,dxo)

            Jy  = 0.0
            gJy = np.zeros(np.shape(x))
            for j in range(0,fdvar.nobstimes):

                i = fdvar.nobstimes - j - 1
                dy = np.dot(H,dxtl[fdvar.twind_obsIndex[i],:]) - d[i,:]
                tint = fdvar.twind[fdvar.twind_obsIndex[i-1]:fdvar.twind_obsIndex[i]+1]

                Jy = Jy + 0.5 * np.dot(np.transpose(dy),np.dot(Rinv,dy))

                gJy = gJy + np.dot(np.transpose(H),np.dot(Rinv,dy))
                if ( len(tint) != 0 ):
                    exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (%f,np.flipud(xnl),fdvar.twind, True))' % (model.Name, model.Par[0]+model.Par[1]))
                    gJy = sxi[-1,:].copy()

            J  =  Jb +  Jy
            gJ = gJb + gJy

            if ( niters == 0 ): print "initial cost = %10.5f" % J
            if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
                print "        cost = %10.5f after %4d iterations" % (J, niters)

            if ( minimization.cg ):
                if ( niters == 0 ):
                    dxo = dxo - alpha_r * gJ
                    cgJold = gJ
                else:
                    beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJold),gJold)
                    cgJ = gJ + beta * cgJold
                    dxo = dxo - alpha_r * cgJ
                    cgJold = cgJ

                gJold = gJ
            else:
                dxo = dxo - alpha_r * gJ

            niters = niters + 1

        print '  final cost = %10.5f after %4d iterations' % (J, niters)

        x = x + dxo

    # advance to the analysis time to get the 4DVAR estimate
    exec('xs = integrate.odeint(%s, x, fdvar.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
    xa = xs[-1,:].copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + Rinv)

    return xa, A, niters
# }}}
###############################################################
