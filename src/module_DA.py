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

    if   ( ensDA.inflation.inflate == 0 ):
        print 'Doing no inflation at all'
    elif ( ensDA.inflation.inflate == 1 ):
        print 'Inflating the Prior using multiplicative inflation with a factor of %f' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.inflate == 2 ):
        print 'Inflating the Prior by adding white-noise with zero-mean and %f spread' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.inflate == 3 ):
        print 'Inflating the Posterior by covariance relaxation method with weight %f to the prior' % ensDA.inflation.infl_fac
    elif ( ensDA.inflation.inflate == 4 ):
        print 'Inflating the Posterior by spread restoration method with a factor of %f' % ensDA.inflation.infl_fac
    else:
        print 'Invalid inflation method'
        print 'ensDA.inflation.inflate must be one of : 0 | 1 | 2 | 3 | 4'
        print 'Multiplicative | Additive | Covariance Relaxation | Spread Restoration'
        fail = True

    if   ( ensDA.localization.localize == 0 ): loc_type = 'No localization'
    elif ( ensDA.localization.localize == 1 ): loc_type = 'Gaspari & Cohn polynomial function'
    elif ( ensDA.localization.localize == 2 ): loc_type = 'Boxcar function'
    elif ( ensDA.localization.localize == 3 ): loc_type = 'Ramped boxcar function'
    else:
        print 'Invalid localization method'
        print 'ensDA.localization.localize must be one of : 0 | 1 | 2 | 3 '
        print 'None | Gaspari & Cohn | Boxcar | Ramped Boxcar'
        loc_type = 'None'
        fail = True
    if ( loc_type != 'None' ):
        print 'Localizing using an %s with a covariance cutoff of %f' % (loc_type, ensDA.localization.cov_cutoff)

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
    if ( (ensDA.inflation.inflate == 1) or (ensDA.inflation.inflate == 2) ):

        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        if   ( ensDA.inflation.inflate == 1 ): # multiplicative inflation
            Xbp = ensDA.inflation.infl_fac * Xbp

        elif ( inflation.inflate == 2 ): # additive white model error (mean:zero, spread:ensDA.inflation.infl_fac)
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
            dist = np.float( np.abs( ob - i ) ) / Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            cov_factor = compute_cov_factor(dist, ensDA.localization)

            temp_ens[i,:] = temp_ens[i,:] + state_inc * cov_factor

    Xa = temp_ens.copy()

    # compute analysis mean and perturbations
    xam = np.mean(Xa,axis=1)
    Xap = np.transpose(np.transpose(Xa) - xam)

    # posterior inflation
    if   ( ensDA.inflation.inflate == 3 ): # covariance relaxation (Zhang & Snyder)
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)
        Xap = Xbp * ensDA.inflation.infl_fac + Xap * (1.0 - ensDA.inflation.infl_fac)

    elif ( ensDA.inflation.inflate == 4 ): # posterior spread restoration (Whitaker & Hamill)
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
def compute_cov_factor(dist, localization):
# {{{
    '''
    compute the covariance factor given distance and localization information

    cov_factor = compute_cov_factor(dist, localization)

          dist - distance between "points"
  localization - localization class
    cov_factor - covariance factor

    localization.localize
        0 : no localization
        1 : Gaspari & Cohn polynomial function
        2 : Boxcar
        3 : Ramped Boxcar
    localization.cov_cutoff
        normalized cutoff distance = cutoff_distance / (2 * normalization_factor)
        Eg. normalized cutoff distance = 1 / (2 * 40)
        localize at 1 point in the 40-variable LE96 model
    '''

    if   ( localization.localize == 0 ): # No localization

        cov_factor = 1.0

    elif ( localization.localize == 1 ): # Gaspari & Cohn localization

        if   ( np.abs(dist) >= 2.0*localization.cov_cutoff ):
            cov_factor = 0.0
        elif ( np.abs(dist) <= localization.cov_cutoff ):
            r = np.abs(dist) / localization.cov_cutoff
            cov_factor = ( ( ( -0.25*r + 0.5 )*r + 0.625 )*r - 5.0/3.0 )*(r**2) + 1.0
        else:
            r = np.abs(dist) / localization.cov_cutoff
            cov_factor = ( ( ( ( r/12 - 0.5 )*r +0.625 )*r + 5.0/3.0 )*r -5.0 )*r + 4.0 - 2.0 / (3.0 * r)

    elif ( localization.localize == 2 ): # Boxcar localization

        if ( np.abs(dist) >= 2.0*localization.cov_cutoff ):
            cov_factor = 0.0
        else:
            cov_factor = 1.0

    elif ( localization.localize == 3 ): # Ramped localization

        if   ( np.abs(dist) >= 2.0*localization.cov_cutoff ):
            cov_factor = 0.0
        elif ( np.abs(dist) <= localization.cov_cutoff ):
            cov_factor = 1.0
        else:
            cov_factor = (2.0 * localization.cov_cutoff - np.abs(dist)) / localization.cov_cutoff

    else:

        print '%d is an invalid localization method' % localization.localize
        sys.exit(1)

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
        if ( varDA.precondition ): print 'Assimilate observations using 3DVar [using incremental formulation] with preconditioning'
        else:                      print 'Assimilate observations using 3DVar [using incremental formulation] without preconditioning'
    elif ( varDA.update == 2 ):
        if ( varDA.precondition ): print 'Assimilate observations using 4DVar [using incremental formulation] with preconditioning'
        else:                      print 'Assimilate observations using 4DVar [using incremental formulation] without preconditioning'
        print 'Assimilate observations using 4DVar [using incremental formulation]'
    else:
        print 'Invalid assimilation algorithm'
        print 'varDA.update must be one of : 0 | 1 | 2'
        print 'No Assimilation | 3DVar | 4DVar'
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

    xa, niters = update_varDA(xb, B, y, R, H, varDA, model=None)

          xb - prior
           B - background error covariance / preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    if   ( varDA.update == 0 ):
        xa, niters = xb, np.NaN

    elif ( varDA.update == 1 ):
        if ( varDA.precondition ): xa, niters = ThreeDvar_pc(xb, B, y, R, H, varDA)
        else:                      xa, niters = ThreeDvar(   xb, B, y, R, H, varDA)

    elif ( varDA.update == 2 ):
        if ( varDA.precondition ): xa, niters = FourDvar_pc(xb, B, y, R, H, varDA, model)
        else:                      xa, niters = FourDvar(   xb, B, y, R, H, varDA, model)

    else:
        print 'invalid update algorithm ...'
        sys.exit(2)

    return xa, niters
# }}}
###############################################################

###############################################################
def ThreeDvar(xb, B, y, R, H, varDA):
# {{{
    '''
    Update the prior with 3Dvar algorithm to produce a posterior.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.

    xa, niters = ThreeDvar(xb, B, y, R, H, varDA)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    # increment  : x - xb = dx
    # innovation :      d = y - H(xb)
    #                  dy = Hdx - d
    # cost function:           J(dx) =  Jb +  Jy
    # cost function gradient: gJ(dx) = gJb + gJy
    #  Jb = 0.5 * dx^T B^{-1} dx
    #  Jy = 0.5 * dy^T R^{-1} dy
    # gJb =     B^{-1} dx
    # gJy = H^T R^{-1} dy
    # gJ  = [ B^{-1} + H^T R^{-1} H ] dx - H^T R^{-1} d

    xa   = xb.copy()
    Binv = np.linalg.inv(B)
    Rinv = np.linalg.inv(R)

    valInd  = np.isfinite(y)

    for outer in range(0,varDA.maxouter):

        d  = y[valInd] - np.dot(H[valInd,:],xa)
        gJ = np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),d))

        dJ      = gJ.copy()
        dx      = np.zeros(np.shape(xa))
        niters  = 0

        residual_first = np.sum(gJ**2)
        residual_tol   = 1.0
        print 'initial residual = %15.10f' % (residual_first)

        while ( (residual_tol >= varDA.minimization.tol) and ( niters <= varDA.minimization.maxiter) ):

            niters = niters + 1

            AdJ = np.dot(Binv,dJ) + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],dJ)))

            [dx, gJ, dJ] = minimize(varDA.minimization, dx, gJ, dJ, AdJ)

            residual     = np.sum(gJ**2)
            residual_tol = residual / residual_first

            if ( not np.mod(niters,5) ):
                print '        residual = %15.10f after %4d iterations' % (residual, niters)

        if ( niters > varDA.minimization.maxiter ): print 'exceeded maximum iterations allowed'
        print '  final residual = %15.10f after %4d iterations' % (residual, niters)

        # 3DVAR estimate
        xa = xa + dx

    return xa, niters
# }}}
###############################################################

###############################################################
def FourDvar(xb, B, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with 4Dvar algorithm to produce a posterior.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.

    xa, niters = FourDvar(xb, B, y, R, H, varDA, model)

          xb - prior
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    # increment  : xo - xbo = dxo
    #                         dxi = Mdxo
    # innovation :      di = yi - H(xbi) = yi - H(M(xbo))
    #                  dyi = Hdxi - di   = HMdxo - di
    # cost function:           J(dxo) =  Jb +  Jy
    # cost function gradient: gJ(dxo) = gJb + gJy
    #  Jb = 0.5 *      dxo^T B^{-1} dxo
    #  Jy = 0.5 * \sum dyi^T R^{-1} dyi
    # gJb =              B^{-1} dxo
    # gJy = \sum M^T H^T R^{-1} dyi
    # gJ  = [ B^{-1} + \sum M^T H^T R^{-1} H M ] dxo - \sum M^T H^T R^{-1} di

    # start with background
    xa   = xb.copy()
    Binv = np.linalg.inv(B)
    Rinv = np.linalg.inv(R)

    for outer in range(0,varDA.maxouter):

        # advance the background through the assimilation window with full non-linear model
        xnl = advance_model(model, xa, varDA.fdvar.twind, perfect=False)

        gJ = np.zeros(np.shape(xb))
        d  = np.zeros(np.shape(y))
        for j in range(0,varDA.fdvar.nobstimes):

            i = varDA.fdvar.nobstimes - j - 1

            valInd = np.isfinite(np.squeeze(y[i,]))

            d[i,:] = y[i,:] - np.dot(H,xnl[varDA.fdvar.twind_obsIndex[i],:])
            gJ = gJ + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),d[i,valInd]))

            tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
            if ( len(tint) != 0 ):
                sxi = advance_model_tlm(model, gJ, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
                gJ = sxi[-1,:].copy()

        dJ     = gJ.copy()
        dxo    = np.zeros(np.shape(xb))
        niters = 0

        residual_first = np.sum(gJ**2)
        residual_tol   = 1.0
        print 'initial residual = %15.10f' % (residual_first)

        while ( (residual_tol >= varDA.minimization.tol) and (niters <= varDA.minimization.maxiter) ):

            niters = niters + 1

            # advance the direction of the gradient through the assimilation window with TL model
            dJtl = advance_model_tlm(model, dJ, varDA.fdvar.twind, xnl, varDA.fdvar.twind, adjoint=False, perfect=False)

            AdJb = np.dot(Binv,dJ)
            AdJy = np.zeros(np.shape(xb))
            for j in range(0,varDA.fdvar.nobstimes):

                i = varDA.fdvar.nobstimes - j - 1

                valInd = np.isfinite(np.squeeze(y[i,]))

                AdJy = AdJy + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],dJtl[varDA.fdvar.twind_obsIndex[i],:])))

                tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
                if ( len(tint) != 0 ):
                    sxi = advance_model_tlm(model, AdJy, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
                    AdJy = sxi[-1,:].copy()

            AdJ = AdJb + AdJy

            [dxo, gJ, dJ] = minimize(varDA.minimization, dxo, gJ, dJ, AdJ)

            residual = np.sum(gJ**2)
            residual_tol = residual / residual_first

            if ( not np.mod(niters,5) ):
                print '        residual = %15.10f after %4d iterations' % (residual, niters)

        if ( niters > varDA.minimization.maxiter ): print 'exceeded maximum iterations allowed'
        print '  final residual = %15.10f after %4d iterations' % (residual, niters)

        xa = xa + dxo

    return xa, niters
# }}}
###############################################################


###############################################################
def ThreeDvar_pc(xb, G, y, R, H, varDA):
# {{{
    '''
    Update the prior with 3Dvar algorithm (with preconditioning) to produce a posterior.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.

    xa, niters = ThreeDvar_pc(xb, G, y, R, H, varDA)

          xb - prior
           G - preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    # increment  : x - xb = dx        = Gw
    # innovation :      d = y - H(xb)
    #                  dy = Hdx - d   = HGw - d
    # cost function:           J(w) =  Jb +  Jy
    # cost function gradient: gJ(w) = gJb + gJy
    #  Jb = 0.5 * w^T w
    #  Jy = 0.5 * dy^T R^{-1} dy
    # gJb = w
    # gJy = G^T H^T R^{-1} dy
    # gJ  = [ I + G^T H^T R^{-1} H G ] w - G^T H^T R^{-1} d

    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    valInd  = np.isfinite(y)

    for outer in range(0,varDA.maxouter):

        d  = y[valInd] - np.dot(H[valInd,:],xa)
        gJ = np.dot(np.transpose(G),np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),d)))

        dJ      = gJ.copy()
        w       = np.zeros(np.shape(xa))
        niters  = 0

        residual_first = np.sum(gJ**2)
        residual_tol   = 1.0
        print 'initial residual = %15.10f' % (residual_first)

        while ( (residual_tol >= varDA.minimization.tol) and ( niters <= varDA.minimization.maxiter) ):

            niters = niters + 1

            AdJ = dJ + np.dot(np.transpose(G),np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],np.dot(G,dJ)))))

            [w, gJ, dJ] = minimize(varDA.minimization, w, gJ, dJ, AdJ)

            residual     = np.sum(gJ**2)
            residual_tol = residual / residual_first

            if ( not np.mod(niters,5) ):
                print '        residual = %15.10f after %4d iterations' % (residual, niters)

        if ( niters > varDA.minimization.maxiter ): print 'exceeded maximum iterations allowed'
        print '  final residual = %15.10f after %4d iterations' % (residual, niters)

        # 3DVAR estimate
        xa = xa + np.dot(G,w)

    return xa, niters
# }}}
###############################################################

###############################################################
def FourDvar_pc(xb, G, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with 4Dvar algorithm (with preconditioning) to produce a posterior.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.

    xa, niters = FourDvar_pc(xb, G, y, R, H, varDA, model)

          xb - prior
           G - preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    # increment  : xo - xbo = dxo                         = Gw
    #                         dxi = Mdxo                  = MGw
    # innovation :      di = yi - H(xbi) = yi - H(M(xbo))
    #                  dyi = Hdxi - di   = HMdxo - di     = HMGw - di
    # cost function:           J(w) =  Jb +  Jy
    # cost function gradient: gJ(w) = gJb + gJy
    #  Jb = 0.5 *      w^T w
    #  Jy = 0.5 * \sum dyi^T R^{-1} dyi
    # gJb = w
    # gJy = \sum G^T M^T H^T R^{-1} dyi
    # gJ  = [ I + \sum G^T M^T H^T R^{-1} H M G ] w - \sum G^T M^T H^T R^{-1} di

    # start with background
    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    for outer in range(0,varDA.maxouter):

        # advance the background through the assimilation window with full non-linear model
        xnl = advance_model(model, xa, varDA.fdvar.twind, perfect=False)

        gJ = np.zeros(np.shape(xb))
        d  = np.zeros(np.shape(y))
        for j in range(0,varDA.fdvar.nobstimes):

            i = varDA.fdvar.nobstimes - j - 1

            valInd = np.isfinite(np.squeeze(y[i,]))

            d[i,:] = y[i,:] - np.dot(H,xnl[varDA.fdvar.twind_obsIndex[i],:])
            gJ = gJ + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),d[i,valInd]))

            tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
            if ( len(tint) != 0 ):
                sxi = advance_model_tlm(model, gJ, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
                gJ = sxi[-1,:].copy()

        gJ = np.dot(np.transpose(G),gJ)

        dJ     = gJ.copy()
        w      = np.zeros(np.shape(xb))
        niters = 0

        residual_first = np.sum(gJ**2)
        residual_tol   = 1.0
        print 'initial residual = %15.10f' % (residual_first)

        while ( (residual_tol >= varDA.minimization.tol) and (niters <= varDA.minimization.maxiter) ):

            niters = niters + 1

            # advance the direction of the gradient through the assimilation window with TL model
            GdJtl = advance_model_tlm(model, np.dot(G,dJ), varDA.fdvar.twind, xnl, varDA.fdvar.twind, adjoint=False, perfect=False)

            AdJb = dJ.copy()
            AdJy = np.zeros(np.shape(xb))
            for j in range(0,varDA.fdvar.nobstimes):

                i = varDA.fdvar.nobstimes - j - 1

                valInd = np.isfinite(np.squeeze(y[i,]))

                AdJy = AdJy + np.dot(np.transpose(H[valInd,:]),np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],GdJtl[varDA.fdvar.twind_obsIndex[i],:])))

                tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
                if ( len(tint) != 0 ):
                    sxi = advance_model_tlm(model, AdJy, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
                    AdJy = sxi[-1,:].copy()

            AdJy = np.dot(np.transpose(G),AdJy)
            AdJ = AdJb + AdJy

            [w, gJ, dJ] = minimize(varDA.minimization, w, gJ, dJ, AdJ)

            residual = np.sum(gJ**2)
            residual_tol = residual / residual_first

            if ( not np.mod(niters,5) ):
                print '        residual = %15.10f after %4d iterations' % (residual, niters)

        if ( niters > varDA.minimization.maxiter ): print 'exceeded maximum iterations allowed'
        print '  final residual = %15.10f after %4d iterations' % (residual, niters)

        xa = xa + np.dot(G,w)

    return xa, niters
# }}}
###############################################################

###############################################################
def minimize(minimization, xo, gJo, dJo, AdJo):
# {{{
    '''
    Perform minimization using conjugate gradient method

    [x, gJ, dJ] = minimize(minimization, xo, gJo, dJo, AdJo)

minimization - minimization class
          xo - quantity to minimize
         gJo - current gradient of the cost function
         dJo - direction of the gradient of the cost function
        AdJo - direction of the gradient of the cost function
    '''

    alpha = np.dot(np.transpose(gJo),gJo) / np.dot(np.transpose(dJo),AdJo)
    x  = xo  + alpha * dJo
    gJ = gJo - alpha * AdJo
    beta = np.dot(np.transpose(gJ),gJ) / np.dot(np.transpose(gJo),gJo)
    dJ = gJ + beta * dJo

    return [x, gJ, dJ]
# }}}
###############################################################

###############################################################
def localization_operator(model, localization):
# {{{
    '''
    Get localization operator given model and localization classes

    L = localization_operator(model, localization)

       model - model class
localization - localization class
           L - localization operator | size(L) == [model.Ndof,model.Ndof]
    '''

    L = np.ones((model.Ndof,model.Ndof))

    for i in range(0,model.Ndof):
        for j in range(0,model.Ndof):
            dist = np.float( np.abs( i - j ) ) / model.Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            L[i,j] = compute_cov_factor(dist, localization)

    return L
# }}}
###############################################################
