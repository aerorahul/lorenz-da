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
import numpy as np
import multiprocessing as mp
###############################################################

###############################################################
class DataAssim(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self):
    #{{{
        '''
        Initializes an empty Data Assimilation class
        '''
        pass
    #}}}

    def init(self,nassim=100,ntimes=0.05,maxouter=1,**kwargs):
    #{{{
        '''
        Populates the Data Assimilation class
          nassim - number of assimilation cycles [100]
          ntimes - interval between assimilation cycles [0.05]
        maxouter - number of outer loops [1]

        Also returns as self.
        t0 - Set start time of assimilation interval to 0.0

        These attributes will be added after initialization:
         Nobs - number of observation in the data-assimilation cycle
        tanal - time vector between two assimilation cycles
        '''

        self.nassim   = nassim
        self.ntimes   = ntimes
        self.maxouter = maxouter

        self.t0       = 0.0

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class VarDataAssim(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self):
    #{{{
        '''
        Initializes an empty Variational Data Assimilation class
        '''
        pass
    #}}}

    def init(self,model,DA, \
            update=1,precondition=1, \
            inflate=1,infl_fac=1.0,infl_adp=False, \
            localize=1,cov_cutoff=0.0625,cov_trunc=None, \
            maxiter=100, tol=1.0e-4, \
            window=0.0,offset=1.0,nobstimes=1, \
            **kwargs):
    #{{{
        '''
        Populates the Variational Data Assimilation class
               model - model class instance
                  DA - Data assimilation class instance
              update - flavour of variational update [1]
                       0 = No update
                       1 = 3DVar
                       2 = 4DVar
        precondition - precondition before update [1]
             inflate - inflate (Bs) static background error cov. [1]
                       0 = no-inflation
                       1 = multiplicative inflation
            infl_fac - inflation factor for inflating Bs [1.0]
            infl_adp - adapt inflation factor for multiple outer loops [False]
            localize - flavour for localizing Bs [1]
                       0 = No localization
                       1 = Gaspari-Cohn localization
                       2 = Box car localization
                       3 = Ramped localization
          cov_cutoff - cutoff for localization [0.0625]
           cov_trunc - truncation for localization [None]
             maxiter - maximum allowed iterations for minimization [100]
                 tol - tolerance to check for convergence in minimization [1.e-4]
              window - length of window between assimilations [0.0]
              offset - start of the window from assimilation time [1.0]
           nobstimes - divide the window in equal obs. bins [1]
        '''

        self.update       = update
        self.precondition = precondition

        self.inflation = inflation(infl_fac=infl_fac,infl_adp=infl_adp)

        if ( cov_trunc is None ): cov_trunc = model.Ndof
        self.localization = localization(localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc)

        self.minimization = minimization(maxiter=maxiter,tol=tol)

        self.fdvar = fdvar(model,DA,window=window,offset=offset,nobstimes=nobstimes)

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class EnsDataAssim(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self):
    #{{{
        '''
        Initializes an empty Ensemble Data Assimilation class
        '''
        pass
    #}}}

    def init(self,model,DA, \
            update=1,Nens=15, init_ens_infl_fac=1.0, \
            inflate=1,infl_fac=1.0, \
            localize=1,cov_cutoff=0.0625,cov_trunc=None, \
            **kwargs):
    #{{{
        '''
        Populates the Ensemble Data Assimilation class
               model - model class instance
                  DA - Data assimilation class instance
              update - flavour of ensemble update [2]
                       0 = No update
                       1 = EnKF
                       2 = EnSRF
                       3 = EAKF
                       4 = ETKF (not implemented yet)
                Nens - number of ensemble members [15]
   init_ens_infl_fac - inflation factor to inflate initial ensemble [1.0]
             inflate - inflation method for sampling error [1]
                       0 = no-inflation
                       1 = multiplicative inflation (scale prior by factor)
                       2 = additive inflation (add white noise [model-error] to prior)
                       3 = covariance relaxation (add scaled prior variance to posterior)
                       4 = spread restoration (scale posterior by a factor based on ratios)
                       5 = adaptive inflation (not implemented yet)
            infl_fac - inflation factor [1.0]
            localize - flavour for localizing Be [1]
                       0 = No localization
                       1 = Gaspari-Cohn localization
                       2 = Box car localization
                       3 = Ramped localization
          cov_cutoff - cutoff for localization [0.0625]
           cov_trunc - truncation for localization [None]
        '''

        self.update = update
        self.Nens   = Nens

        self.init_ens_infl_fac = init_ens_infl_fac

        self.inflation = inflation(infl_fac=infl_fac)

        if ( cov_trunc is None ): cov_trunc = model.Ndof
        self.localization = localization(localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc)

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class minimization(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self,maxiter=100,tol=1.0e-4,**kwargs):
    # {{{
        '''
        Initializes the minimization class
        maxiter - maximum iterations allowed for minimization [100]
            tol - tolerance to check for convergence of minimization [1.e-4]
        '''

        self.maxiter = maxiter
        self.tol     = tol

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class inflation(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self,inflate=1,infl_fac=1.0,infl_adp=False,**kwargs):
    #{{{
        '''
        Initializes the inflation class
         inflate - inflation method for inflating Bs / sampling error [1]
                   0 = no-inflation
                   1 = multiplicative inflation (scale Bs / prior ensemble by factor)
                   2 = additive inflation (add white noise [model-error] to prior ensemble)
                   3 = covariance relaxation (add scaled prior ensemble variance to posterior)
                   4 = spread restoration (scale posterior ensemble variance by a factor based on ratios)
                   5 = adaptive inflation (not implemented yet)
        infl_fac - inflation factor [1.0]
        infl_adp - adaptive inflation factor for Bs for multiple outer loops [False]
        '''

        self.inflate  = inflate
        self.infl_fac = infl_fac
        self.infl_adp = infl_adp

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class localization(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self,localize=1,cov_cutoff=0.0625,cov_trunc=40,**kwargs):
    #{{{
        '''
        Initializes the localization class
          localize - flavour for localizing Be [1]
                     0 = No localization
                     1 = Gaspari-Cohn localization
                     2 = Box car localization
                     3 = Ramped localization
        cov_cutoff - cutoff for localization [0.0625]
         cov_trunc - truncation for localization [None]
        '''

        self.localize   = localize
        self.cov_cutoff = cov_cutoff
        self.cov_trunc  = cov_trunc

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

class fdvar(object):

    def __setattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val

    def __delattr__(self,key,val):
        if key in self.__dict__:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]

    def __init__(self,model,DA,window=0.0,offset=1.0,nobstimes=1,**kwargs):
    #{{{
        '''
        Initializes the 4DVar class
            model - model class instance
               DA - Data assimilation class instance
           window - length of window between assimilations [0.0]
           offset - start of the window from assimilation time [1.0]
        nobstimes - divide the window in equal obs. bins [1]

        Also returns as self.
        tb - time index of background from previous analysis
        ta - time index of analysis from previous analysis
        tf - time index of end of window from previous analysis
        tw - difference of indices between tf and tb

        tbkgd - time vector to background from previous analysis
        tanal - time vector to analysis from new background
        tfore - time vector to end of window from previous analysis
        twind - time vector from beginning of window to end of window

        twind_obsInterval - interval between obs. in terms of time-steps
           twind_obsTimes - time vector containing indices of obs. in window
           twind_obsIndex - interval between obs. in terms of time-steps in terms of time-steps
        '''

        self.window    = window
        self.offset    = offset
        self.nobstimes = nobstimes

        # check length of assimilation window
        if ( self.offset + self.window < 1.0 ):
            raise ValueError('Assimilation window is too short')

        # time index from analysis to ... background, next analysis, end of window, window
        self.tb = np.int(np.rint(self.offset * DA.ntimes/model.dt))
        self.ta = np.int(np.rint(DA.ntimes/model.dt))
        self.tf = np.int(np.rint((self.offset + self.window) * DA.ntimes/model.dt))
        self.tw = self.tf - self.tb

        # time vector from analysis to ... background, next analysis, end of window, window
        self.tbkgd = np.linspace(DA.t0,self.tb,                self.tb+1) * model.dt
        self.tanal = np.linspace(DA.t0,self.ta-self.tb,self.ta-self.tb+1) * model.dt
        self.tfore = np.linspace(DA.t0,self.tf,                self.tf+1) * model.dt
        self.twind = np.linspace(DA.t0,self.tw,                self.tw+1) * model.dt

        # time vector, interval, indices of observations
        if   ( self.nobstimes == 1 ):
            self.twind_obsInterval = 0
            self.twind_obsTimes    = self.twind.copy()

        elif ( self.nobstimes  > 1 ):
            self.twind_obsInterval = self.tw / (self.nobstimes-1)
            self.twind_obsTimes    = self.twind[::self.twind_obsInterval]

        self.twind_obsIndex = np.array(np.rint(self.twind_obsTimes / model.dt), dtype=int)
        self.twind_obs      = np.linspace(DA.t0,self.twind_obsInterval,self.twind_obsInterval+1) * model.dt

        overlap = np.int(self.tf - (1.0+self.offset)*self.ta)
        if ( overlap >= 0.0 ):
            self.noverlap = np.sum(i <= overlap for i in self.twind_obsIndex)
        else:
            self.noverlap = 0

        for key, value in kwargs.items(): self.__setattr__(key,value)
    #}}}

def check_DA(DA):
# {{{
    '''
    Check for valid DA options

    check_DA(DA)

    DA - data assimilation class
    '''

    print('===========================================')

    fail = False

    print('Cycle DA for %d cycles' % DA.nassim)
    print('Interval between each DA cycle is %f' % DA.ntimes)
    if ( hasattr(DA,'do_hybrid') ):
        if ( DA.do_hybrid ):
            print('Doing hybrid data assimilation')
            print('Using %d%% of the flow-dependent covariance' % (np.int(DA.hybrid_wght * 100)))
            if ( DA.hybrid_rcnt ): print('Re-centering the ensemble about the central analysis')
            else:                  print('No re-centering of the ensemble about the central analysis')

    print('===========================================')

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def check_ensDA(DA,ensDA):
# {{{
    '''
    Check for valid ensemble DA algorithms and methods

    check_ensDA(DA,ensDA)

       DA - data assimilation class
    ensDA - ensemble data assimilation class
    '''

    check_DA(DA)

    print('===========================================')

    fail = False

    if   ( ensDA.update == 0 ):
        print('Running "No Assimilation"')
    elif ( ensDA.update == 1 ):
        print('Assimilate observations using the EnKF')
    elif ( ensDA.update == 2 ):
        print('Assimilate observations using the EnSRF')
    elif ( ensDA.update == 3 ):
        print('Assimilate observations using the EAKF')
    else:
        print('Invalid assimilation algorithm')
        print('ensDA.update must be one of : 0 | 1 | 2 | 3')
        print('No Assimilation | EnKF | EnSRF | EAKF')
        fail = True

    if   ( ensDA.inflation.inflate == 0 ):
        print('Doing no inflation at all')
    elif ( ensDA.inflation.inflate == 1 ):
        print('Inflating the Prior using multiplicative inflation with a factor of %f' % ensDA.inflation.infl_fac)
    elif ( ensDA.inflation.inflate == 2 ):
        print('Inflating the Prior by adding white-noise with zero-mean and %f spread' % ensDA.inflation.infl_fac)
    elif ( ensDA.inflation.inflate == 3 ):
        print('Inflating the Posterior by covariance relaxation method with weight %f to the prior' % ensDA.inflation.infl_fac)
    elif ( ensDA.inflation.inflate == 4 ):
        print('Inflating the Posterior by spread restoration method with a factor of %f' % ensDA.inflation.infl_fac)
    else:
        print('Invalid inflation method')
        print('ensDA.inflation.inflate must be one of : 0 | 1 | 2 | 3 | 4')
        print('Multiplicative | Additive | Covariance Relaxation | Spread Restoration')
        fail = True

    if   ( ensDA.localization.localize == 0 ): loc_type = 'No localization'
    elif ( ensDA.localization.localize == 1 ): loc_type = 'Gaspari & Cohn polynomial function'
    elif ( ensDA.localization.localize == 2 ): loc_type = 'Boxcar function'
    elif ( ensDA.localization.localize == 3 ): loc_type = 'Ramped boxcar function'
    else:
        print('Invalid localization method')
        print('ensDA.localization.localize must be one of : 0 | 1 | 2 | 3 ')
        print('None | Gaspari & Cohn | Boxcar | Ramped Boxcar')
        loc_type = 'None'
        fail = True
    if ( loc_type != 'None' ):
        print('Localizing using an %s with a covariance cutoff of %f' % (loc_type, ensDA.localization.cov_cutoff))

    print('===========================================')

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_ensDA(Xb, y, R, H, ensDA, model):
# {{{
    '''
    Update the prior with an ensemble-based state estimation algorithm to produce a posterior

    Xa, A, error_variance_ratio = update_ensDA(Xb, B, y, R, H, ensDA, model)

          Xb - prior ensemble
           y - observations
           R - observation error covariance
           H - forward operator
       ensDA - ensemble data assimilation class
       model - model class
          Xa - posterior ensemble
     evratio - ratio of innovation variance to total variance
    '''

    # prior inflation
    if ( (ensDA.inflation.inflate == 1) or (ensDA.inflation.inflate == 2) ):

        xbm = np.mean(Xb,axis=1)
        Xbp = (Xb.T - xbm).T

        if   ( ensDA.inflation.inflate == 1 ): # multiplicative inflation
            Xbp = ensDA.inflation.infl_fac * Xbp

        elif ( ensDA.inflation.inflate == 2 ): # additive white model error (mean:zero, spread:ensDA.inflation.infl_fac)
            Xbp = Xbp + inflation.infl_fac * np.random.randn(model.Ndof,ensDA.Nens)

        Xb = (Xbp.T + xbm).T

    temp_ens = Xb.copy()

    # initialize innovation and total variance
    innov  = np.zeros(y.shape[0]) * np.NaN
    totvar = np.zeros(y.shape[0]) * np.NaN

    # assimilate all obs., one-by-one
    for ob in range(y.shape[0]):

        if ( np.isnan(y[ob]) ): continue

        ye = np.dot(H[ob,:],temp_ens)

        if   ( ensDA.update == 0 ): # no assimilation
            obs_inc, innov[ob], totvar[ob] = np.zeros(model.Ndof), np.NaN, np.NaN

        elif ( ensDA.update == 1 ): # update using the EnKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnKF(y[ob], R[ob,ob], ye)

        elif ( ensDA.update == 2 ): # update using the EnSRF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EnSRF(y[ob], R[ob,ob], ye)

        elif ( ensDA.update == 3 ): # update using the EAKF
            obs_inc, innov[ob], totvar[ob] = obs_increment_EAKF(y[ob], R[ob,ob], ye)

        else:
            print('invalid update algorithm ...')
            sys.exit(2)

        for i in range(model.Ndof):
            state_inc = state_increment(obs_inc, temp_ens[i,:], ye)

            # localization
            dist = np.float( np.abs( ob - i ) ) / model.Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            cov_factor = compute_cov_factor(dist, ensDA.localization)

            temp_ens[i,:] = temp_ens[i,:] + state_inc * cov_factor

    Xa = temp_ens.copy()

    # compute analysis mean and perturbations
    xam = np.mean(Xa,axis=1)
    Xap = (Xa.T - xam).T

    # posterior inflation
    if   ( ensDA.inflation.inflate == 3 ): # covariance relaxation (Zhang & Snyder)
        xbm = np.mean(Xb,axis=1)
        Xbp = (Xb.T - xbm).T
        Xap = Xbp * ensDA.inflation.infl_fac + Xap * (1.0 - ensDA.inflation.infl_fac)

    elif ( ensDA.inflation.inflate == 4 ): # posterior spread restoration (Whitaker & Hamill)
        xbs = np.std(Xb,axis=1,ddof=1)
        xas = np.std(Xa,axis=1,ddof=1)
        for i in range(model.Ndof):
            Xap[i,:] =  np.sqrt((ensDA.inflation.infl_fac * (xbs[i] - xas[dof])/xas[i]) + 1.0) * Xap[i,:]

    # add inflated perturbations back to analysis mean
    Xa = (Xap.T + xam).T

    # check for filter divergence
    error_variance_ratio = np.nansum(innov**2) / np.nansum(totvar)
    if ( 0.5 < error_variance_ratio < 2.0 ):
        print('total error / total variance = %f' % (error_variance_ratio))
    else:
        print("\033[0;31mtotal error / total variance = %f | WARNING : filter divergence\033[0m" % (error_variance_ratio))
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

        print('%d is an invalid localization method' % localization.localize)
        sys.exit(1)

    return cov_factor
# }}}
###############################################################

###############################################################
def check_varDA(DA,varDA):
# {{{
    '''
    Check for valid variational DA algorithms

    check_varDA(DA,varDA)

       DA - data assimilation class
    varDA - variational data assimilation class
    '''

    check_DA(DA)

    print('===========================================')

    fail = False

    if   ( varDA.precondition == 0 ): pstr = 'no'
    elif ( varDA.precondition == 1 ): pstr = 'square-root B'
    elif ( varDA.precondition == 2 ): pstr = 'full B'

    if   ( varDA.update == 0 ):
        print('Running "No Assimilation"')
    elif ( varDA.update == 1 ):
        print('Assimilate observations using 3DVar [using incremental formulation] with %s preconditioning'% (pstr))
    elif ( varDA.update == 2 ):
        print('Assimilate observations using 4DVar [using incremental formulation] with %s preconditioning'% (pstr))
    else:
        print('Invalid assimilation algorithm')
        print('varDA.update must be one of : 0 | 1 | 2')
        print('No Assimilation | 3DVar | 4DVar')
        fail = True

    if   ( varDA.inflation.inflate ):
        print('Inflating the static background error covariance with a factor of %f' % varDA.inflation.infl_fac)
    else:
        print('Doing no inflation of the static background error covariance at all')
    print('===========================================')

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_varDA(xb, B, y, R, H, varDA, model):
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

    elif ( varDA.update in [1,2] ):
        xa, niters = VarSolver(xb, B, y, R, H, varDA, model)

    else:
        print('invalid update algorithm ...')
        sys.exit(2)

    return xa, niters
# }}}
###############################################################

###############################################################
def VarSolver(xb, B, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with Variational algorithm to produce a posterior.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.

    xa, niters = VarSolver(xb, B, y, R, H, varDA, model)

          xb - prior
           B - Background error covariance / preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function

    NO PRECONDITIONING
    increment  : xo - xbo = dxo
                            dxi = Mdxo
    innovation :      di = yi - H(xbi) = yi - H(M(xbo))
                     dyi = Hdxi - di   = HMdxo - di
    cost function:           J(dxo) =  Jb +  Jy
    cost function gradient: gJ(dxo) = gJb + gJy
     Jb = 0.5 *      dxo^T B^{-1} dxo
     Jy = 0.5 * \sum dyi^T R^{-1} dyi
    gJb =              B^{-1} dxo
    gJy = \sum M^T H^T R^{-1} dyi
    gJ  = [ B^{-1} + \sum M^T H^T R^{-1} H M ] dxo - \sum M^T H^T R^{-1} di

    PRECONDITION WITH sqrt(B) = G
    increment  : xo - xbo = dxo                         = Gw
                            dxi = Mdxo                  = MGw
    innovation :      di = yi - H(xbi) = yi - H(M(xbo))
                     dyi = Hdxi - di   = HMdxo - di     = HMGw - di
    cost function:           J(w) =  Jb +  Jy
    cost function gradient: gJ(w) = gJb + gJy
     Jb = 0.5 *      w^T w
     Jy = 0.5 * \sum dyi^T R^{-1} dyi
    gJb = w
    gJy = \sum G^T M^T H^T R^{-1} dyi
    gJ  = [ I + \sum G^T M^T H^T R^{-1} H M G ] w - \sum G^T M^T H^T R^{-1} di

    PRECONDITION WITH B
    increment  : xo - xbo = dxo                         = Bw
                            dxi = Mdxo
    innovation :      di = yi - H(xbi) = yi - H(M(xbo))
                     dyi = Hdxi - di   = HMdxo - di
    cost function:           J(w) =  Jb +  Jy
    cost function gradient: gJ(w) = gJb + gJy
     Jb = 0.5 *      w^T w
     Jy = 0.5 * \sum dyi^T R^{-1} dyi
    gJb = w
    gJy = \sum M^T H^T R^{-1} dyi
    gJ  = w + \sum M^T H^T R^{-1} H M dxo - \sum M^T H^T R^{-1} di
    '''

    # start with background
    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    # advance the background through the assimilation window with full non-linear model
    xnl = model.advance(xa, varDA.fdvar.twind, perfect=False)

    g = np.zeros(xa.shape)
    d = np.zeros( y.shape)

    for j in range(varDA.fdvar.nobstimes):

        i = varDA.fdvar.nobstimes - j - 1

        valInd = np.isfinite(y[i,])

        d[i,:] = y[i,:] - np.dot(H,xnl[varDA.fdvar.twind_obsIndex[i],:])

        g = g + Jgrad(H[valInd,:], np.diag(Rinv[valInd,valInd]), d[i,valInd])

        tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
        if ( len(tint) != 0 ):
            sxi = model.advance_tlm(g, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
            g = sxi[-1,:].copy()

    if   ( varDA.precondition == 0 ):
        r = g.copy()
        s = 0.0
        p = r.copy()
        q = 0.0
        v = np.zeros(r.shape)
        w = 0.0
    elif ( varDA.precondition == 1 ):
        r = np.dot(B.T,g)
        s = 0.0
        p = r.copy()
        q = 0.0
        v = 0.0
        w = np.zeros(r.shape)
    elif ( varDA.precondition == 2 ):
        r = -g
        s = np.dot(B,r)
        p = -s
        q = -r
        v = np.zeros(r.shape)
        w = np.zeros(s.shape)

    niters = 0

    residual_first = np.sum(r**2+s**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and (niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        if   ( varDA.precondition == 0 ): tmp = p.copy()
        elif ( varDA.precondition == 1 ): tmp = np.dot(B,p)
        elif ( varDA.precondition == 2 ): tmp = p.copy()

        # advance the direction of the gradient through the assimilation window with TL model
        tmptl = model.advance_tlm(tmp, varDA.fdvar.twind, xnl, varDA.fdvar.twind, adjoint=False, perfect=False)

        Ap = np.zeros(xb.shape)

        for j in range(varDA.fdvar.nobstimes):

            i = varDA.fdvar.nobstimes - j - 1

            valInd = np.isfinite(y[i,])

            Ap = Ap + hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), tmptl[varDA.fdvar.twind_obsIndex[i],:])

            tint = varDA.fdvar.twind[varDA.fdvar.twind_obsIndex[i-1]:varDA.fdvar.twind_obsIndex[i]+1]
            if ( len(tint) != 0 ):
                sxi = model.advance_tlm(Ap, tint, xnl, varDA.fdvar.twind, adjoint=True, perfect=False)
                Ap = sxi[-1,:].copy()

        if   ( varDA.precondition == 0 ): Ap = np.dot(np.linalg.inv(B),p) + Ap
        elif ( varDA.precondition == 1 ): Ap = p + np.dot(B.T,Ap)
        elif ( varDA.precondition == 2 ): Ap = q + Ap

        [v,w,r,s,p,q] = minimize(varDA,v,w,r,s,p,q,Ap,B)

        residual     = np.sum(r**2+s**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    # Variational estimate
    if   ( varDA.precondition == 0 ): xa = xa + v
    elif ( varDA.precondition == 1 ): xa = xa + np.dot(B,w)
    elif ( varDA.precondition == 2 ): xa = xa + v

    return xa, niters
# }}}
###############################################################

###############################################################
def ThreeDvar_adj(gradJ, B, y, R, H, varDA, model):
# {{{
    '''
    This attempts to mimic the adjoint for the ThreeDvar to
    compute the observation impact.
    A step in observation impact calculation is K^T gradJ
    where K^T is the adjoint of the Kalman gain and
    gradJ is the gradient of the metric at analysis time.
    K         =          [ B^{-1} + H^T R^{-1} H ]^{-1} H^T R^{-1}
    K^T       = R^{-1} H [ B^{-1} + H^T R^{-1} H ]^{-1}
    K^T gradJ = R^{-1} H [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ
    In this routine we solve for q:
    [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ = q
    i.e.
    [ B^{-1} + H^T R^{-1} H ] q          = gradJ

    and then obtain K^T gradJ as R^{-1} H q

    This is only valid for a single outer loop. Multiple outer loops
    need the innovations saved for each outer loop during forward analysis.

    The interface is kept the same as that of ThreeDvar for
    simplicity

    KTgradJ, niters = ThreeDvar_adj(gradJ, B, y, R, H, varDA, model)

       gradJ - model sensitivity gradient
           B - background error covariance
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
     KTgradJ - K^T gradJ
      niters - number of iterations required for minimizing the Hessian
    '''

    # KTgradJ = K^T gradJ
    #         = R^{-1} H [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ
    #         = R^{-1} H q
    # where
    # [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ = q
    # i.e.
    # [ B^{-1} + H^T R^{-1} H ]          q = gradJ

    # Solving Ax = b
    # where
    # A = [ B^{-1} + H^T R^{-1} H ]
    # x = q
    # b = gradJ
    # KTgradJ = R^{-1} H q

    Binv = np.linalg.inv(B)
    Rinv = np.linalg.inv(R)

    valInd = np.isfinite(y)

    gJ     = gradJ.copy()
    dJ     = gJ.copy()
    q      = np.zeros(gJ.shape)
    niters = 0

    residual_first = np.sum(gJ**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and ( niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        AdJ = np.dot(Binv,dJ) + hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), dJ)

        [q, gJ, dJ] = minimize(q, gJ, dJ, AdJ)

        residual     = np.sum(gJ**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    KTgradJ = np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],q))

    return KTgradJ, niters
# }}}
###############################################################

###############################################################
def ThreeDvar_pc_adj(gradJ, G, y, R, H, varDA, model):
# {{{
    '''
    This attempts to mimic the adjoint for the ThreeDvar_pc to
    compute the observation impact.
    In this implementation, the incremental form is used.
    It is the same as the classical formulation.
    A step in observation impact calculation is K^T gradJ
    where K^T is the adjoint of the Kalman gain and
    gradJ is the gradient of the metric at analysis time.
    K         =          G [ I + G^T H^T R^{-1} H G]^{-1} G^T H^T R^{-1}
    K^T       = R^{-1} H [ B^{-1} + H^T R^{-1} H ]^{-1}
    K^T gradJ = R^{-1} H [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ
    In this routine we solve for q:
    [ B^{-1} + H^T R^{-1} H ]^{-1} gradJ = q
    i.e.
    [ B^{-1} + H^T R^{-1} H ] q          = gradJ

    and then obtain K^T gradJ as R^{-1} H G q

    This is only valid for a single outer loop. Multiple outer loops
    need the innovations saved for each outer loop during forward analysis.

    The interface is kept the same as that of ThreeDvar_pc for
    simplicity

    KTgradJ, niters = ThreeDvar_pc_adj(gradJ, G, y, R, H, varDA, model)

       gradJ - model sensitivity gradient
           G - preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
     KTgradJ - K^T gradJ
      niters - number of iterations required for minimizing the Hessian

    KTgradJ = K^T gradJ
            = R^{-1} H G [ I + G^T H^T R^{-1} H G]^{-1} G^T gradJ
            = R^{-1} H G [ I + G^T H^T R^{-1} H G]^{-1} w
            = R^{-1} H G q
    where
    w = G^T gradJ
    therefore
    [ I + G^T H^T R^{-1} H G]^{-1} w = q
    i.e.
    [ I + G^T H^T R^{-1} H G]      q = w

    Solving Ax = b
    where
    A = [ I + G^T H^T R^{-1} H G]
    x = q
    b = w
    KTgradJ = R^{-1} H G q
    '''

    Rinv = np.linalg.inv(R)

    valInd  = np.isfinite(y)

    w      = np.dot(G.T,gradJ)
    gJ     = w.copy()
    dJ     = gJ.copy()
    q      = np.zeros(gJ.shape)
    niters = 0

    residual_first = np.sum(gJ**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and ( niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        AdJ = dJ + np.dot(G.T, hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), np.dot(G,dJ)))

        [q, gJ, dJ] = minimize(q, gJ, dJ, AdJ)

        residual     = np.sum(gJ**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    KTgradJ = np.dot(np.diag(Rinv[valInd,valInd]),np.dot(H[valInd,:],np.dot(G,q)))

    return KTgradJ, niters
# }}}
###############################################################

###############################################################
def check_hybDA(DA,ensDA,varDA):
# {{{
    '''
    Check for valid hybrid DA algorithms

    check_hybDA(DA,ensDA,varDA)

       DA - data assimilation class
    ensDA - ensemble data assimilation class
    varDA - variational data assimilation class
    '''

    check_DA(DA)
    check_ensDA(DA,ensDA)
    check_varDA(DA,varDA)

    return
# }}}
###############################################################

###############################################################
def check_ensvarDA(DA,ensDA,varDA):
# {{{
    '''
    Check for valid ensemble-variational DA algorithms

    check_ensvarDA(DA,ensDA,varDA)

       DA - data assimilation class
    ensDA - ensemble data assimilation class
    varDA - variational data assimilation class
    '''

    check_hybDA(DA,ensDA,varDA)

    fail = False

    print('===========================================')

    if   ( varDA.precondition == 0 ): pstr = 'nothing'
    elif ( varDA.precondition == 1 ): pstr = 'square-root B'
    elif ( varDA.precondition == 2 ): pstr = 'full B'

    if   ( varDA.precondition != 1 ):
        print('''Preconditioning with %s is not allowed when
assimilating observations using ensemble-variational algorithm.
varDA.precondition must be : 1 = square-root B''' % pstr)
        fail = True

    print('===========================================')

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_ensvarDA(xb, D, S, y, R, H, varDA, ensDA, model):
# {{{
    '''
    Update the prior with a ensemble-variational-based state estimation algorithm to produce a posterior
    This algorithm is implemented with the control-vector formulation

    xa, niters = update_ensvarDA(xb, Xb, S, y, R, H, varDA, ensDA, model)

          xb - prior
           D - diagonalized ensemble prior within the window
           S - localization matrix [full or sqrt]
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       ensDA - ensemble data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    if   ( varDA.update == 0 ):
        xa, niters = xb, np.NaN

    elif ( varDA.update in [1,2] ):
        xa, niters = EnsembleVarSolver(xb, S, D, y, R, H, varDA, ensDA, model)

    else:
        print('invalid update algorithm ...')
        sys.exit(2)

    return xa, niters
# }}}
###############################################################

###############################################################
def EnsembleVarSolver(xb, S, D, y, R, H, varDA, ensDA, model):
# {{{
    '''
    Update the prior with Ensemble-based Variational algorithm to produce a posterior.
    This implementation uses the alpha-control vector for minimization to find the weights
    to the ensemble. The alpha here is denoted by v.

    xa, niters = EnsembleVarSolver(xb, S, D, y, R, H, varDA, model)

          xb - prior
           S - localization matrix [full or sqrt]
           D - diagonalized ensemble prior in the window
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       ensDA - ensemble data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function

    NO PRECONDITIONING [Solve using Conjugate Gradient]
    [ S^{-1} + \sum Di^T Hi^T Ri^{-1} Hi Di ] v = \sum Di^T Hi^T Ri^{-1} di
    solve for v

    PRECONDITION WITH sqrt(S) = G [Solve using Conjugate Gradient]
    [ I + \sum G^T Di^T Hi^T Ri^{-1} Hi Di G ] G^{-1} v = \sum G^T Di^T Hi^T Ri^{-1} di
    let w = G^{-1} v
    [ I + \sum G^T Di^T Hi^T Ri^{-1} Hi Di G ] w        = \sum G^T Di^T Hi^T Ri^{-1} di
    solve for w
    v = G w

    PRECONDITION WITH S [Solve using Double Conjugate Gradient]
    [ S^{-1} + \sum Di^T Hi^T Ri^{-1} Hi Di ] v = \sum Di^T Hi^T Ri^{-1} di
    let w = S^{-1} v
    solve for both v and w

    analysis increment : dxi = Di v
    '''

    # start with background
    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    # advance the background through the assimilation window with full non-linear model
    xnl = model.advance(xa, varDA.fdvar.twind, perfect=False)

    d = np.zeros(y.shape)
    g = np.zeros(ensDA.Nens*model.Ndof)

    for i in range(varDA.fdvar.nobstimes):

        valInd = np.isfinite(y[i,])

        d[i,:] = y[i,:] - np.dot(H,xnl[varDA.fdvar.twind_obsIndex[i],:])

        g = g + np.dot(D[i,:,:].T, Jgrad(H[valInd,:], np.diag(Rinv[valInd,valInd]), d[i,valInd]))

    if   ( varDA.precondition == 0 ):
        r = g.copy()
        s = 0.0
        p = r.copy()
        q = 0.0
        v = np.zeros(r.shape)
        w = 0.0
    elif ( varDA.precondition == 1 ):
        r = np.dot(S.T,g)
        s = 0.0
        p = r.copy()
        q = 0.0
        v = 0.0
        w = np.zeros(r.shape)
    elif ( varDA.precondition == 2 ):
        r = -g
        s = np.dot(S,r)
        p = -s
        q = -r
        v = np.zeros(r.shape)
        w = np.zeros(s.shape)

    niters = 0

    residual_first = np.sum(r**2+s**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and (niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        if   ( varDA.precondition == 0 ): tmp = p.copy()
        elif ( varDA.precondition == 1 ): tmp = np.dot(S,p)
        elif ( varDA.precondition == 2 ): tmp = p.copy()

        Ap = np.zeros(tmp.shape)

        for j in range(varDA.fdvar.nobstimes):

            i = varDA.fdvar.nobstimes - j - 1

            valInd = np.isfinite(y[i,])

            Ap = Ap + np.dot(D[i,:,:].T, hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), np.dot(D[i,:,:],tmp)))

        if   ( varDA.precondition == 0 ): Ap = np.dot(np.linalg.inv(S),p) + Ap
        elif ( varDA.precondition == 1 ): Ap = p + np.dot(S.T,Ap)
        elif ( varDA.precondition == 2 ): Ap = q + Ap

        [v,w,r,s,p,q] = minimize(varDA,v,w,r,s,p,q,Ap,S)

        residual     = np.sum(r**2+s**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    if   ( varDA.precondition == 0 ): dxo = np.dot(D[0,:,:],v)
    elif ( varDA.precondition == 1 ): dxo = np.dot(D[0,:,:],np.dot(S,w))
    elif ( varDA.precondition == 2 ): dxo = np.dot(D[0,:,:],v)

    xa = xa + dxo

    return xa, niters
# }}}
###############################################################

###############################################################
def EnsembleVar(xb, G, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with Ensemble-based Variational algorithm to produce a posterior.
    This algorithm utilizes the sqrt(B) preconditioning described in Buehner 2005.
    The sqrt(B) used here is the localized ensemble matrix
    Although no longer used, this piece of code is left for legacy reasons to test
    any new implementations.

    xa, niters = EnsembleVar(xb, G, y, R, H, varDA, ensDA, model)

          xb - prior
           G - preconditioning matrix
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       ensDA - ensemble data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function

    increment  : xo - xbo = dxo                         = Gw
                            dxi = Mdxo                  = MGw
    innovation :      di = yi - H(xbi) = yi - H(M(xbo))
                     dyi = Hdxi - di   = HMdxo - di     = HMGw - di
    cost function:           J(w) =  Jb +  Jy
    cost function gradient: gJ(w) = gJb + gJy
     Jb = 0.5 *      w^T w
     Jy = 0.5 * \sum dyi^T R^{-1} dyi
    gJb = w
    gJy = \sum [HMG]^T R^{-1} dyi
    gJ  = [ I + \sum [HMG]^T R^{-1} [HMG] ] w - \sum [HMG]^T R^{-1} di
    '''

    # start with background
    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    # advance the background through the assimilation window with full non-linear model
    xnl = model.advance(xa, varDA.fdvar.twind, perfect=False)

    d  = np.zeros(y.shape)
    g  = np.zeros(ensDA.Nens*varDA.localization.cov_trunc)

    for i in range(varDA.fdvar.nobstimes):

        valInd = np.isfinite(y[i,])

        d[i,:] = y[i,:] - np.dot(H,xnl[varDA.fdvar.twind_obsIndex[i],:])

        g = g + np.dot(G[i,:,:].T, Jgrad(H[valInd,:], np.diag(Rinv[valInd,valInd]), d[i,valInd]))

    r  = g.copy()
    s  = 0.0
    p  = r.copy()
    q  = 0.0
    dx = 0.0
    w  = np.zeros(r.shape)

    niters = 0

    residual_first = np.sum(r**2+s**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and (niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        Ap = p.copy()

        for i in range(varDA.fdvar.nobstimes):

            valInd = np.isfinite(y[i,])

            Ap = Ap + np.dot(G[i,:,:].T, hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), np.dot(G[i,:,:],p)))

        [dx,w,r,s,p,q] = minimize(varDA,dx,w,r,s,p,q,Ap,G)

        residual = np.sum(r**2+s**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    xa = xa + np.dot(G[0,:,:],w)

    return xa, niters
# }}}
###############################################################

###############################################################
def check_hybensvarDA(DA,ensDA,varDA):
# {{{
    '''
    Check for valid hybrid ensemble-variational DA algorithms

    check_hybensvarDA(DA,ensDA,varDA)

       DA - data assimilation class
    ensDA - ensemble data assimilation class
    varDA - variational data assimilation class
    '''

    check_hybDA(DA,ensDA,varDA)

    fail = False

    print('===========================================')

    if   ( varDA.precondition == 0 ): pstr = 'nothing'
    elif ( varDA.precondition == 1 ): pstr = 'square-root B'
    elif ( varDA.precondition == 2 ): pstr = 'full B'

    if   ( varDA.precondition == 0 ):
        print('''Preconditioning with %s is not allowed when
assimilating observations using hybrid ensemble-variational algorithm.
Inverting B = [[Bs 0], [0 A]] is not possible, as it is singular.
varDA.precondition must be : 1 | 2 = square-root or full B''' % pstr)
        fail = True

    print('===========================================')

    if ( fail ): sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def update_hybensvarDA(xb, B, C, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with a hybrid ensemble-variational-based state estimation algorithm
    to produce a posterior.
    This algorithm is implemented with the extended control vector method of Lorenc 03.

    xa, niters = update_hybensvarDA(xb, B, C, y, R, H, varDA, model)

          xb - prior
           B - (scaled) static background error cov. and localization corr. matrix [[Bs 0],[0 A]]
           C - preconditioned identity and ensemble matrix = [I D]
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
        xa, niters = HybridEnsembleThreeDvar(xb, B, C, y, R, H, varDA, model)

    elif ( varDA.update == 2 ):
        xa, niters = HybridEnsembleFourDvar(xb, B, C, y, R, H, varDA, model)

    else:
        print('invalid update algorithm ...')
        sys.exit(2)

    return xa, niters
# }}}
###############################################################

###############################################################
def HybridEnsembleThreeDvar(xb, B, C, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with Hybrid Ensemble-based 3Dvar algorithm to produce a posterior.
    This algorithm is implemented with the extended control vector method of Lorenc 03.

    xa, niters = HybridEnsembleThreeDvar(xb, B, C, y, R, H, varDA, model)

          xb - prior
           B - (scaled) static background error cov. and localization corr. matrix [[Bs 0],[0 A]]
           C - preconditioned identity and ensemble matrix = [I D]
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    xa   = xb.copy()
    Rinv = np.linalg.inv(R)

    valInd = np.isfinite(y)

    d = y[valInd] - np.dot(H[valInd,:],xa)

    g = np.dot(C.T, Jgrad(H[valInd,:], np.diag(Rinv[valInd,valInd]), d))

    if   ( varDA.precondition == 1 ):
        r  = np.dot(B.T,g)
        s  = 0.0
        p  = r.copy()
        q  = 0.0
        dx = 0.0
        w  = np.zeros(r.shape)
    elif ( varDA.precondition == 2 ):
        r  = -g
        s  = np.dot(B,r)
        p  = -s
        q  = -r
        dx = np.zeros(B.shape[0])
        w  = np.zeros(B.shape[0])

    niters = 0

    residual_first = np.sum(r**2+s**2)
    residual_tol   = 1.0
    print('initial residual = %15.10f' % (residual_first))

    while ( (np.sqrt(residual_tol) >= varDA.minimization.tol**2) and ( niters <= varDA.minimization.maxiter) ):

        niters = niters + 1

        if   ( varDA.precondition == 1 ): tmp = np.dot(B,p)
        elif ( varDA.precondition == 2 ): tmp = p.copy()

        Ap = np.dot(C.T, hessian(H[valInd,:], np.diag(Rinv[valInd,valInd]), np.dot(C,tmp)))

        if   ( varDA.precondition == 1 ): Ap = p + np.dot(B.T,Ap)
        elif ( varDA.precondition == 2 ): Ap = q + Ap

        [dx,w,r,s,p,q] = minimize(varDA,dx,w,r,s,p,q,Ap,B)

        residual     = np.sum(r**2+s**2)
        residual_tol = residual / residual_first

        if ( not np.mod(niters,5) ):
            print('        residual = %15.10f after %4d iterations' % (residual, niters))

    if ( niters > varDA.minimization.maxiter ): print('\033[0;31mexceeded maximum iterations allowed\033[0m')
    print('  final residual = %15.10f after %4d iterations' % (residual, niters))

    # 3DVAR estimate
    if   ( varDA.precondition == 1 ): xa = xa + np.dot(C,np.dot(B,w))
    elif ( varDA.precondition == 2 ): xa = xa + np.dot(C,dx)

    return xa, niters
# }}}
###############################################################

###############################################################
def HybridEnsembleFourDvar(xb, B, C, y, R, H, varDA, model):
# {{{
    '''
    Update the prior with Hybrid Ensemble-based 4Dvar algorithm to produce a posterior.
    This algorithm is implemented with the extended control vector method of Lorenc 03.

    xa, niters = HybridEnsembleFourDvar(xb, B, C, y, R, H, varDA, model)

          xb - prior
           B - (scaled) static background error cov. and localization corr. matrix [[Bs 0],[0 A]]
           C - preconditioned identity and ensemble matrix = [I D]
           y - observations
           R - observation error covariance
           H - forward operator
       varDA - variational data assimilation class
       model - model class
          xa - posterior
      niters - number of iterations required for minimizing the cost function
    '''

    return xa, niters
# }}}
###############################################################

###############################################################
def Jgrad(H,Rinv,d):
# {{{
    '''
    Compute the initial cost function gradient H^T R^{-1} d

    Jgrad = Jgrad(H,Rinv,d)

       H - forward operator
    Rinv - observation error covariance inverse
       d - innovation vector
    '''

    Jgrad = np.dot(H.T,np.dot(Rinv,d))

    return Jgrad
# }}}
###############################################################

###############################################################
def hessian(H,Rinv,q):
# {{{
    '''
    Compute the Hessian operator H^T R^{-1} H q

    hessian = hessian(H,Rinv,d)

       H - forward operator
    Rinv - observation error covariance inverse
       q - input vector
    '''

    hessian = np.dot(H.T,np.dot(Rinv,np.dot(H,q)))

    return hessian
# }}}
###############################################################

###############################################################
def minimize(varDA,dxo,wo,ro,so,po,qo,Apo,B):
# {{{
    '''
    Call appropriate minimization method

    [dx,w,r,s,p,q] = minimize(dxo,wo,ro,so,po,qo,Apo,B)

  varDA - variational data assimilation class
 dxo,wo - initial iterates to minimize
  ro,so - initial residual of the cost function
  po,qo - initial search direction of the gradient of the cost function
    Apo - A times search direction
      B - preconditioning matrix
    '''

    if   ( varDA.precondition == 0 ):
        dx,r,p = cg(dxo,ro,po,Apo)
        w = wo ; s = so; q = qo
    elif ( varDA.precondition == 1 ):
        w,r,p = cg(wo,ro,po,Apo)
        dx = dxo ; s = so; q = qo
    elif ( varDA.precondition == 2 ):
        dx,w,r,s,p,q = doublecg(dxo,wo,ro,so,po,qo,Apo,B)

    return [dx,w,r,s,p,q]
# }}}
###############################################################

###############################################################
def cg(xo, ro, do, Ado):
# {{{
    '''
    Perform minimization using conjugate gradient method

    [x, r, d] = cg(xo, ro, do, Ado)

     xo - initial iterate to minimize
     ro - initial residual of the cost function
     do - initial search direction of the gradient of the cost function
    Ado - A times search direction
    '''

    alpha = np.dot(ro.T,ro) / np.dot(do.T,Ado)
    x = xo + alpha * do
    r = ro - alpha * Ado
    beta = np.dot(r.T,r) / np.dot(ro.T,ro)
    d = r + beta * do

    return [x, r, d]
# }}}
###############################################################

###############################################################
def doublecg(wo,zo,ro,so,po,qo,Apo,B):
# {{{
    '''
    Perform minimization using double conjugate gradient method

    [w,z,r,s,p,q] = doublecg(wo,zo,ro,so,po,qo,Apo,B)

   wo,zo - initial iterates to minimize
  ro, so - initial residuals of the cost function
  po, qo - initial search directions of the gradient of the cost function
     Apo - A times search direction
       B - preconditioning matrix
    '''

    alpha = np.dot(ro.T,so) / np.dot(po.T,Apo)
    w = wo + alpha * po
    z = zo + alpha * qo
    r = ro + alpha * Apo
    s = np.dot(B,r)
    beta = np.dot(r.T,s) / np.dot(ro.T,so.T)
    p = beta * po - s
    q = beta * qo - r

    return [w,z,r,s,p,q]
# }}}
###############################################################

###############################################################
def precondition(X, varDA, ensDA, model, L=None):
# {{{
    '''
    Setup the preconditioner before variational data assimilation

    G = precondition(X, varDA, ensDA, model, L=None)

     X - matrix to precondition
 varDA - variational-based data assimilation class
 ensDA - ensemble-based data assimilation class
 model - minimization class
     L - localize the matrix to precondition, if desired [None]
    '''

    if   ( varDA.update == 1 ):

        Xp = (X[0,:,:].T - np.mean(X[0,:,:],axis=1)).T
        if ( L is None ):
            G = Xp.copy()
        else:
            G = np.zeros((model.Ndof,varDA.localization.cov_trunc*ensDA.Nens))
            for m in range(ensDA.Nens):
                si = varDA.localization.cov_trunc *  m
                ei = varDA.localization.cov_trunc * (m+1)
                G[:,si:ei] = np.dot(np.diag(Xp[:,m]),L) / np.sqrt(ensDA.Nens - 1.0)

    elif ( varDA.update == 2 ):

        if ( L is None ): G = np.zeros(X.shape)
        else:             G = np.zeros((varDA.fdvar.nobstimes,model.Ndof,varDA.localization.cov_trunc*ensDA.Nens))

        for i in range(varDA.fdvar.nobstimes):
            Xp = (X[i,:,:].T - np.mean(X[i,:,:],axis=1)).T
            if ( L is None ):
                G[i,:,:] = Xp.copy()
            else:
                for m in range(ensDA.Nens):
                    si = varDA.localization.cov_trunc *  m
                    ei = varDA.localization.cov_trunc * (m+1)
                    G[i,:,si:ei] = np.dot(np.diag(Xp[:,m]),L) / np.sqrt(ensDA.Nens - 1.0)

    return G
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

    for i in range(model.Ndof):
        for j in range(model.Ndof):
            dist = np.float( np.abs( i - j ) ) / model.Ndof
            if ( dist > 0.5 ): dist = 1.0 - dist
            L[i,j] = compute_cov_factor(dist, localization)

    return L
# }}}
###############################################################

###############################################################
def advance_ensemble(Xi, t, model, perfect=True, parallel=False, **kwargs):
# {{{
    '''
    Advance an ensemble given initial conditions, length of integration and model information.

    Xf = advance_ensemble(Xi, T, model, perfect=True, parallel=False, **kwargs)

       Xi - Ensemble of initial conditions; size(Xi) = [N == Ndof, M == Nens]
        t - integrate from t[0] to t[end]
    model - model class
  perfect - If perfect model run for L96, use model.Par[0], else use model.Par[1]
 parallel - perform model advance in parallel on multiple processors
 **kwargs - any additional arguments that need to go in the model advance call
       Xf - Ensemble of final states; size(Xf) = [N == Ndof, M == Nens]
    '''

    Xf = np.zeros(Xi.shape)

    if ( parallel ):

        result_queue = mp.Queue()
        madvs = [ model.advance(xi,t,perfect=perfect,result=result_queue) for xi in Xi.T ]
        jobs = [ mp.Process(madv) for madv in madvs ]
        for job in jobs: job.start()
        for job in jobs: job.join()
        Xs = [ result_queue.get() for madv in madvs ]
        for m, xs in enumerate(Xs):
            Xf[:,m] = xs[-1,:].copy()

    else:

        for m, xi in enumerate(Xi.T):
            xs = model.advance(xi, t, perfect=perfect, **kwargs)
            Xf[:,m] = xs[-1,:].copy()

    return Xf
# }}}
###############################################################

###############################################################
def inflate_ensemble(Xi, inflation_factor):
# {{{
    '''
    Inflate an ensemble.

    Xo = inflate_ensemble(Xi, inflation_factor)

              Xi - Input ensemble    [ shape(Xi) = [Ne, Ndof] ]
inflation_factor - Factor with which to inflate ensemble perturbations
              Xo - Inflated ensemble [ shape(Xo) = [Ne, Ndof] ]
    '''

    xm = np.mean(Xi,axis=0)
    Xo = xm + inflation_factor * (Xi - xm)

    return Xo
# }}}
###############################################################

###############################################################
def compute_B(varDA,Bc,outer=0):
# {{{
    '''
    Load climatological background error covariance matrix and
    and make ready for variational update

    B = compute_B(varDA,Bc,outer=0)

    Bc - climatological background error covariance
 varDA - variational-based data assimilation class
 outer - outer loop index, if adaptively scaling Bc (0)
     B - background error covariance matrix / preconditioning matrix
    '''

    # inflate climatological background error cov. matrix
    B = varDA.inflation.infl_fac * Bc
    if ( varDA.inflation.infl_adp ): B /= ( outer + 1 )

    # precondition B to sqrt(B)
    if ( varDA.precondition == 1 ):
        [U,S2,_] = np.linalg.svd(B, full_matrices=True, compute_uv=True)
        B = np.dot(U,np.diag(np.sqrt(S2)))

    return B
# }}}
###############################################################

###############################################################
def create_obs(model,varDA,xt,H,R,yold=None):
# {{{
    '''
    y = create_obs(model,varDA,xt,H,R,yold=None)

    Create observations for EnKF / 3DVar / 4DVar within a specified obs. window

 model - model class
 varDA - ensemble / variational-based data assimilation class
    xt - truth
     H - forward operator
     R - observation error covariance
     y - observation vector / matrix
  yold - observation vector / matrix from previous cycle [None]
    '''
    # new observations from noise about truth

    if ( not hasattr(varDA,'minimization') ):
        # This is an ensemble update
        y = np.zeros((1,model.Ndof))
        y[0,:] = np.dot(H,xt + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
        return y

    if ( varDA.update in [1,2] ):

        y = np.zeros((varDA.fdvar.nobstimes,model.Ndof))

        # integrate truth within the obs. window and collect state at obs. times
        xs = model.advance(xt, varDA.fdvar.twind, perfect=True)[varDA.fdvar.twind_obsIndex,:]

        for i in range(varDA.fdvar.nobstimes):
            if ( i < varDA.fdvar.noverlap ):
                y[i,:] = yold[varDA.fdvar.nobstimes-varDA.fdvar.noverlap+i,:].copy()
            else:
                y[i,:] = np.dot(H,xs[i,:] + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))

        return y

    raise ValueError('create_obs should never reach here.')
# }}}
###############################################################
