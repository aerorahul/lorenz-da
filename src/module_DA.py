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
import sys
import numpy         as     np
from   scipy         import integrate
from   module_Lorenz import *
###############################################################

###############################################################
def check_DA(DA):
# {{{
    '''
    Check for valid DA options

    check_DA(DA)

    DA - data assimilation class
    '''

    print '==========================================='

    fail = False

    print 'Cycle DA for %d cycles' % DA.nassim
    print 'Interval between each DA cycle is %f' % DA.ntimes
    if ( hasattr(DA,'do_hybrid') ):
        if ( DA.do_hybrid ):
            print 'Doing hybrid data assimilation'
            print 'Using %d%% of the flow-dependent covariance' % (np.int(DA.hybrid_wght * 100))
            if ( DA.hybrid_rcnt ): print 'Re-centering the ensemble about the central analysis'
            else:                  print 'No re-centering of the ensemble about the central analysis'

    print '==========================================='

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def check_ensDA(ensDA):
# {{{
    '''
    Check for valid ensemble DA algorithms and methods

    check_ensDA(ensDA)

    ensDA - ensemble data assimilation class
    '''

    print '==========================================='

    fail = False

    if   ( ensDA.update == 0 ):
        print 'Running "No Assimilation"'
    elif ( ensDA.update == 1 ):
        print 'Assimilate observations using the EnKF'
    elif ( ensDA.update == 2 ):
        print 'Assimilate observations using the EnSRF'
    elif ( ensDA.update == 3 ):
        print 'Assimilate observations using the EAKF'
    else:
        print 'Invalid assimilation algorithm'
        print 'ensDA.update must be one of : 0 | 1 | 2 | 3'
        print 'No Assimilation | EnKF | EnSRF | EAKF'
        fail = True

    if   ( ensDA.inflation.infl_meth == 1 ):
        print 'Inflating the Prior using multiplicative inflation with a factor of %f' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.infl_meth == 2 ):
        print 'Inflating the Prior by adding white-noise with zero-mean and %f spread' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.infl_meth == 3 ):
        print 'Inflating the Posterior by covariance relaxation method with weight %f to the prior' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.infl_meth == 4 ):
        print 'Inflating the Posterior by spread restoration method with a factor of %f' % ensDA.inflation.infl_fac
    else:
        print 'Invalid inflation method'
        print 'ensDA.inflation.infl_meth must be one of : 1 | 2 | 3 | 4'
        print 'Multiplicative | Additive | Covariance Relaxation | Spread Restoration'
        fail = True

    if   ( ensDA.localization.localize == True ):
        print 'Localizing using Gaspari-Cohn with a covariance cutoff of %f' % ensDA.localization.cov_cutoff
    else:
        print 'No localization'

    print '==========================================='

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_ensDA(Xb, y, R, H, ensDA):
# {{{
    '''
    Update the prior with an ensemble-based state estimation algorithm to produce a posterior

    Xa, A, error_variance_ratio = update_ensDA(Xb, B, y, R, H, ensDA)

          Xb - prior ensemble
           y - observations
           R - observation error covariance
           H - forward operator
       ensDA - ensemble data assimilation class
          Xa - posterior ensemble
     evratio - ratio of innovation variance to total variance
    '''

    Nobs = np.shape(y)[0]
    Ndof = np.shape(Xb)[0]
    Nens = np.shape(Xb)[1]

    innov  = np.zeros(Nobs) * np.NaN
    totvar = np.zeros(Nobs) * np.NaN

    # prior inflation
    if ( (ensDA.inflation.infl_meth == 1) or (ensDA.inflation.infl_meth == 2) ):

        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        if   ( ensDA.inflation.infl_meth == 1 ): # multiplicative inflation
            Xbp = ensDA.inflation.infl_fac * Xbp

        elif ( inflation.infl_meth == 2 ): # additive white model error (mean:zero, spread:ensDA.inflation.infl_fac)
            Xbp = Xbp + inflation.infl_fac * np.random.randn(Ndof,Nens)

        Xb = np.transpose(np.transpose(Xbp) + xbm)

    temp_ens = Xb.copy()

    for ob in range(0, Nobs):

        if ( np.isnan(y[ob]) ): continue

        ye = np.dot(H[ob,:],temp_ens)

        if   ( ensDA.update == 0 ): # no assimilation
            obs_inc, innov[ob], totvar[ob] = np.zeros(Ndof), np.NaN, np.NaN

        elif ( ensDA.update == 1 ): # update using the EnKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnKF(y[ob], R[ob,ob], ye)

        elif ( ensDA.update == 2 ): # update using the EnSRF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnSRF(y[ob], R[ob,ob], ye)

        elif ( ensDA.update == 3 ): # update using the EAKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EAKF(y[ob], R[ob,ob], ye)

        else:
            print 'invalid update algorithm ...'
            sys.exit(2)

        for i in range(0,Ndof):
            state_inc = state_increment(obs_inc, temp_ens[i,:], ye)

            # localization
            if ( ensDA.localization.localize ):
                dist = np.abs( ob - i ) / Ndof
                if ( dist > 0.5 ): dist = 1.0 - dist
                cov_factor = compute_cov_factor(dist, ensDA.localization.cov_cutoff)
            else:
                cov_factor = 1.0

            temp_ens[i,:] = temp_ens[i,:] + state_inc * cov_factor

    Xa = temp_ens.copy()

    # compute analysis mean and perturbations
    xam = np.mean(Xa,axis=1)
    Xap = np.transpose(np.transpose(Xa) - xam)

    # posterior inflation
    if   ( ensDA.inflation.infl_meth == 3 ): # covariance relaxation (Zhang & Snyder)
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)
        Xap = Xbp * ensDA.inflation.infl_fac + Xap * (1.0 - ensDA.inflation.infl_fac)

    elif ( ensDA.inflation.infl_meth == 4 ): # posterior spread restoration (Whitaker & Hamill)
        xbs = np.std(Xb,axis=1,ddof=1)
        xas = np.std(Xa,axis=1,ddof=1)
        for i in range(0,Ndof):
            Xap[i,:] =  np.sqrt((ensDA.inflation.infl_fac * (xbs[i] - xas[dof])/xas[i]) + 1.0) * Xap[i,:]

    # add inflated perturbations back to analysis mean
    Xa = np.transpose(np.transpose(Xap) + xam)

    # check for filter divergence
    error_variance_ratio = np.nansum(innov**2) / np.nansum(totvar)
    if ( 0.5 < error_variance_ratio < 2.0 ):
        print 'total error / total variance = %f' % (error_variance_ratio)
    else:
        print 'total error / total variance = %f | WARNING : filter divergence' % (error_variance_ratio)
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

    valInd  = np.isfinite(y)

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

        dy  = np.dot(H[valInd,:],x) - y[valInd]
        Jy  = 0.5 * np.dot(np.transpose(dy),np.dot(np.diag(Rinv[valInd,valInd]),dy))
        gJy = np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),dy))

        J  =  Jb +  Jy
        gJ = gJb + gJy

        if ( niters == 0 ): print 'initial cost = %10.5f' % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        if ( niters == 0 ) :
            gJold  = gJ
            cgJold = gJ

        [x, gJold, cgJold] = minimize(minimization, niters, minimization.alpha, x, gJ, gJold, cgJold)

        niters = niters + 1

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # 3DVAR estimate
    xa = x.copy()

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),H[valInd,:])))

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

    valInd  = np.isfinite(y)

    # start with zero analysis increment
    dx      = np.zeros(np.shape(xb))
    d       = y[valInd] - np.dot(H[valInd,:],xb)
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
        Jy = 0.5 * np.dot(np.transpose(np.dot(H[valInd,:],dx)-d),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],dx)-d))
        J = Jb + Jy

        gJb = np.dot(Binv,dx)
        gJy = np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],dx)-d))
        gJ  = gJb + gJy

        if ( niters == 0 ): print 'initial cost = %10.5f' % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        if ( niters == 0 ):
            gJold  = 0
            cgJold = 0

        [dx, gJold, cgJold] = minimize(minimization, niters, minimization.alpha, dx, gJ, gJold, cgJold)

        niters = niters + 1

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # 3DVAR estimate
    xa = xb + dx

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),H[valInd,:])))

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
    xold    = xb.copy()
    J       = 1e5
    Jmin    = 1e5
    Jold    = 1e6
    gJold   = 0
    cgJold  = 0
    Binv    = np.linalg.inv(B)
    Rinv    = np.linalg.inv(R)
    niters  = 0
    alpha_r = minimization.alpha

    while ( np.abs(Jold - J) > minimization.tol ):

        if ( niters > minimization.maxiter ):
            print 'exceeded maximum iterations allowed'
            break

        Jold = J

        # advance the background through the assimilation window with the full non-linear model
        if   ( model.Name == 'L63'):
            exec('xnl = integrate.odeint(%s, x, fdvar.twind, (model.Par,0.0))' % (model.Name))
        elif ( model.Name == 'L96'):
            exec('xnl = integrate.odeint(%s, x, fdvar.twind, (model.Par[0]+model.Par[1],0.0))' % (model.Name))
        else:
            print 'model %s is not defined' % model.Name
            sys.exit(2)

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

            valInd = np.isfinite(np.squeeze(y[i,]))

            dy = np.dot(H[valInd,:],xnl[fdvar.twind_obsIndex[i],:]) - y[i,valInd]

            Jy  = Jy  + 0.5 * np.dot(np.transpose(dy),np.dot(np.diag(Rinv[valInd,valInd]),dy))
            gJy = gJy + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),dy))

            tint = fdvar.twind[fdvar.twind_obsIndex[i-1]:fdvar.twind_obsIndex[i]+1]

            if ( len(tint) != 0 ):
                if   ( model.Name == 'L63' ):
                    exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (model.Par,np.flipud(xnl),fdvar.twind, True))' % (model.Name))
                elif ( model.Name == 'L96' ):
                    exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (model.Par[0]+model.Par[1],np.flipud(xnl),fdvar.twind, True))' % (model.Name))
                gJy = sxi[-1,:].copy()

        J  =  Jb +  Jy
        gJ = gJb + gJy

        if ( niters == 0 ): print 'initial cost = %10.5f' % J
        if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
            print '        cost = %10.5f after %4d iterations' % (J, niters)

        # if the cost function increased, reset x and cut line-search parameter by half
        if ( J > Jold ):
            print 'decreasing alpha ...'
            x              = xold.copy()
            alpha_r        = 0.5 * alpha_r
            J              = 1e5
            Jold           = 1e6
            niters         = niters - 1
            increase_alpha = False
        else:
            increase_alpha = True

        # try to increase alpha, if we are past the difficult part
        if ( ( increase_alpha ) and ( alpha_r < minimization.alpha ) ):
            print 'increasing alpha ...'
            alpha_r = 1.1 * alpha_r

        # keep a copy of x, incase cost function increases in the next step
        xold = x.copy()

        [x, gJold, cgJold] = minimize(minimization, niters, alpha_r, x, gJ, gJold, cgJold)

        niters = niters + 1

        # save cost function minima, in case next step is to larger value
        if ( J < Jmin ):
            Jmin = J
            xa   = x.copy()

    print '  final cost = %10.5f after %4d iterations' % (J, niters)

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + np.dot(np.transpose(H),np.dot(Rinv,H)))

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
    xa   = xb.copy()
    d    = np.zeros(np.shape(y))
    Binv = np.linalg.inv(B)
    Rinv = np.linalg.inv(R)

    for outer in range(0,fdvar.maxouter):

        dxo     = np.zeros(np.shape(xb))
        dxold   = np.zeros(np.shape(xb))
        J       = 1e5
        Jmin    = 1e5
        Jold    = 1e6
        gJold   = 0
        cgJold  = 0
        niters  = 0
        alpha_r = minimization.alpha

        # advance the background through the assimilation window with full non-linear model
        if   ( model.Name == 'L63'):
            exec('xnl = integrate.odeint(%s, xa, fdvar.twind, (model.Par,0.0))' % (model.Name))
        elif ( model.Name == 'L96'):
            exec('xnl = integrate.odeint(%s, xa, fdvar.twind, (model.Par[0]+model.Par[1],0.0))' % (model.Name))
        else:
            print 'model %s is not defined' % model.Name
            sys.exit(2)

        # get observational increment at all observation times
        for i in range(0,fdvar.nobstimes):
            d[i,:] = y[i,:] - np.dot(H,xnl[fdvar.twind_obsIndex[i],:])

        while ( np.abs(Jold - J) > minimization.tol ):

            if ( niters > minimization.maxiter ):
                print 'exceeded maximum iterations allowed'
                break

            Jold = J

            # advance the increment through the assimilation window with TL model
            if   ( model.Name == 'L63'):
                exec('dxtl = integrate.odeint(%s_tlm, dxo, fdvar.twind, (model.Par,xnl,fdvar.twind,False))' % (model.Name))
            elif ( model.Name == 'L96'):
                exec('dxtl = integrate.odeint(%s_tlm, dxo, fdvar.twind, (model.Par[0]+model.Par[1],xnl,fdvar.twind,False))' % (model.Name))

            # cost function : J(dx_o) = Jb + Jy
            # Jb = 0.5 * [dx_o]^T B^(-1) [dx_o]
            # Jy = 0.5 * \sum [Hdx_i-d_i]^T R^(-1) [Hdx_i-d_i]
            # cost function gradient: gJ = gJb + gJy
            # gJb = B^(-1) [dx_o]
            # gJy = \sum M^T H^T R^(-1) [Hdx_i-d_i]

            Jb  = 0.5 * np.dot(np.transpose(dxo),np.dot(Binv,dxo))
            gJb = np.dot(Binv,dxo)

            Jy  = 0.0
            gJy = np.zeros(np.shape(xb))
            for j in range(0,fdvar.nobstimes):

                i = fdvar.nobstimes - j - 1

                valInd = np.isfinite(np.squeeze(y[i,]))

                dy = np.dot(H[valInd,:],dxtl[fdvar.twind_obsIndex[i],:]) - d[i,valInd]

                Jy  = Jy  + 0.5 * np.dot(np.transpose(dy),np.dot(np.diag(Rinv[valInd,valInd]),dy))
                gJy = gJy + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),dy))

                tint = fdvar.twind[fdvar.twind_obsIndex[i-1]:fdvar.twind_obsIndex[i]+1]

                if ( len(tint) != 0 ):
                    if   ( model.Name == 'L63'):
                        exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (model.Par,np.flipud(xnl),fdvar.twind, True))' % (model.Name))
                    elif   ( model.Name == 'L96'):
                        exec('sxi = integrate.odeint(%s_tlm, gJy, tint, (model.Par[0]+model.Par[1],np.flipud(xnl),fdvar.twind, True))' % (model.Name))
                    gJy = sxi[-1,:].copy()

            J  =  Jb +  Jy
            gJ = gJb + gJy

            if ( niters == 0 ): print 'initial cost = %10.5f' % J
            if ( ( not np.mod(niters,10) ) and ( not niters == 0 ) ):
                print '        cost = %10.5f after %4d iterations' % (J, niters)

            # if the cost function increased, reset dxo and cut line-search parameter by half
            if ( J > Jold ):
                print 'decreasing alpha ...'
                dxo            = dxold.copy()
                alpha_r        = 0.5 * alpha_r
                J              = 1e5
                Jold           = 1e6
                niters         = niters - 1
                increase_alpha = False
            else:
                increase_alpha = True

            # try to increase alpha, if we are past the difficult part
            if ( ( increase_alpha ) and ( alpha_r < minimization.alpha ) ):
                print 'increasing alpha ...'
                alpha_r = 1.1 * alpha_r

            # keep a copy of dxo, incase cost function increases in the next step
            dxold = dxo.copy()

            [dxo, gJold, cgJold] = minimize(minimization, niters, alpha_r, dxo, gJ, gJold, cgJold)

            niters = niters + 1

            # save cost function minima, in case next step is to larger value
            if ( J < Jmin ):
                Jmin  = J
                dxmin = dxo.copy()

        print '  final cost = %10.5f after %4d iterations' % (J, niters)

        xa = xa + dxmin

    # analysis error covariance from Hessian
    A = np.linalg.inv(Binv + np.dot(np.transpose(H),np.dot(Rinv,H)))

    return xa, A, niters
# }}}
###############################################################

###############################################################
def minimize(minimization, iteration, dstep, x, gJ, gJold, cgJold):
# {{{
    '''
    Perform minimization using steepest descent / conjugate gradient method

    [x, gJold, cgJold] = minimize(minimization, niters, x, gJ, gJold, cgJold)

minimization - minimization class
   iteration - iteration number ( required for conj. grad. method )
       dstep - size of step in the direction of the gradient
           x - quantity to minimize
          gJ - current gradient of the cost function
       gJold - previous gradient of the cost function ( required for conj. grad. method )
      cgJold - previous conjugate gradient of the cost function ( required for conj. grad. method )
    '''

    if ( minimization.cg ):
        if ( iteration == 0 ): # first iteration do a line search
            x = x - dstep * gJ
            cgJold = gJ
        else:
            beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJold),gJold)
            cgJ = gJ + beta * cgJold
            x = x - dstep * cgJ
            cgJold = cgJ

        gJold = gJ
    else:
        x = x - dstep * gJ
        gJold  = None
        cgJold = None

    return [x, gJold, cgJold]
# }}}
###############################################################
