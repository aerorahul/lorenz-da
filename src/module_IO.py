#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# module_IO.py - Functions related to IO for data assimilation
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import os
import sys
import numpy   as     np
from   netCDF4 import Dataset
###############################################################

###############################################################
module = 'module_IO.py'
###############################################################

###############################################################
def create_diag(fname, fattr, ndof, nobs=None, nens=None, hybrid=False):
# {{{
    '''
    create an output file for writing diagnostics

    create_diag(fname, ndof, nens=None, nobs=None)

    fname - name of the output file
    fattr - global attributes
     ndof - number of degrees of freedom in the model
     nobs - number of observations (None)
     nens - number of ensemble members (None)
   hybrid - flag for hybrid DA (False)
    '''

    source = 'create_diag'

    if ( nobs == None ):
        nobs = ndof

    if ( ( hybrid ) and ( nens == None ) ):
        print 'nens cannot be None if doing hybrid'
        sys.exit(2)

    try:

        nc  = Dataset(fname, mode='w', clobber=True, format='NETCDF4')

        Dim = nc.createDimension('ntime',size=None)
        Dim = nc.createDimension('ndof', size=ndof)
        Dim = nc.createDimension('nobs', size=nobs)

        if not ( nens == None ):
            Dim = nc.createDimension('ncopy',size=nens)

        Var = nc.createVariable('truth','f8',('ntime','ndof',))

        if ( nens == None ):
            Var = nc.createVariable('prior',    'f8',('ntime','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','ndof',))
            Var = nc.createVariable('niters',   'f8',('ntime',))
        else:
            Var = nc.createVariable('prior',    'f8',('ntime','ncopy','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','ncopy','ndof',))
            Var = nc.createVariable('evratio',  'f8',('ntime',))

        Var = nc.createVariable('obs',         'f8',('ntime','nobs',))
        Var = nc.createVariable('obs_operator','f8',('ntime','nobs','ndof',))
        Var = nc.createVariable('obs_err_var', 'f8',('ntime','nobs',))

        if ( hybrid ):
            Var = nc.createVariable('prior_emean',    'f8',('ntime','ndof',))
            Var = nc.createVariable('posterior_emean','f8',('ntime','ndof',))
            Var = nc.createVariable('niters',         'f8',('ntime',))

        for (key,value) in fattr.iteritems():
            exec( 'nc.%s = %s' % (key,value) )

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during creating  %s' % (fname)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_diag(fname, time, truth, prior, posterior, obs, obs_operator, obs_err_var, prior_emean=None, posterior_emean=None, niters=None, evratio=None):
# {{{
    '''
    write the diagnostics to an output file

    write_diag(fname, time, truth, prior, posterior, obs, obs_err_var)

              fname - name of the output file, must already exist
               time - time index to write diagnostics for
              truth - truth
              prior - prior state
          posterior - posterior state
                obs - observations
       obs_operator - forward observation operator
        obs_err_var - observation error variance
        prior_emean - prior ensemble mean (None)
    posterior_emean - posterior ensemble mean (None)
             niters - no. of iterations for 3/4DVar to converge (None)
            evratio - error-to-variance ration (None)
    '''

    source = 'write_diag'

    if not os.path.isfile(fname):
        print 'file does not exist ' + fname
        sys.exit(2)

    try:

        nc = Dataset(fname, mode='a', clobber=True, format='NETCDF4')

        nc.variables['truth'][time,:] = truth.copy()

        if ( len(np.shape(prior)) == 1 ):
            nc.variables['prior'][time,:]   = prior.copy()
        else:
            nc.variables['prior'][time,:,:] = prior.copy()

        if ( len(np.shape(posterior)) == 1 ):
            nc.variables['posterior'][time,:]   = posterior.copy()
        else:
            nc.variables['posterior'][time,:,:] = posterior.copy()

        nc.variables['obs'][time,:]          = obs.copy()
        nc.variables['obs_operator'][time,:] = obs_operator.copy()
        nc.variables['obs_err_var'][time,:]  = obs_err_var.copy()

        if not ( prior_emean == None ):
            nc.variables['prior_emean'][time,:] = prior_emean.copy()

        if not ( posterior_emean == None ):
            nc.variables['posterior_emean'][time,:] = posterior_emean.copy()

        if not ( niters == None ):
            nc.variables['niters'][time] = niters

        if not ( evratio == None ):
            nc.variables['evratio'][time] = evratio

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during writing to %s' % (fname)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def read_diag(fname, time, end_time=None):
# {{{
    '''
    read the diagnostics from an output file given name and time index

    read_diag(fname, time, end_time=None)

              fname - name of the file to read from, must already exist
               time - time index to read diagnostics
           end_time - return chunk of data from time to end_time (None)
              truth - truth
              prior - prior state
          posterior - posterior state
                obs - observations
       obs_operator - forward observation operator
        obs_err_var - observation error variance
        prior_emean - observation error variance ( if doing hybrid )
    posterior_emean - observation error variance ( if doing hybrid )
    '''

    source = 'read_diag'

    if not os.path.isfile(fname):
        print 'file does not exist ' + fname
        sys.exit(2)

    if ( end_time == None ): end_time = time + 1

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        truth        = np.squeeze(nc.variables['truth'][time:end_time,])
        prior        = np.squeeze(nc.variables['prior'][time:end_time,])
        posterior    = np.squeeze(nc.variables['posterior'][time:end_time,])
        obs          = np.squeeze(nc.variables['obs'][time:end_time,])
        obs_operator = np.squeeze(nc.variables['obs_operator'][time:end_time,])
        tmp          = np.squeeze(nc.variables['obs_err_var'][time:end_time,])

        if ( end_time - time == 1 ):
            obs_err_var = np.diag(tmp)
        else:
            obs_err_var = np.zeros((np.shape(tmp)[0],np.shape(tmp)[1],np.shape(tmp)[1]))
            for k in range(0, np.shape(tmp)[0]):
                obs_err_var[k,] = np.diag(tmp[k,])

        if 'do_hybrid' in nc.ncattrs():
            hybrid = nc.do_hybrid
        else:
            hybrid = False

        if ( hybrid ):
            prior_mean     = np.squeeze(nc.variables['prior_emean'][time:end_time,])
            posterior_mean = np.squeeze(nc.variables['posterior_emean'][time:end_time,])

        if 'niters' in nc.variables.keys():
            niters = np.squeeze(nc.variables['niters'][time:end_time])

        if 'evratio' in nc.variables.keys():
            evratio = np.squeeze(nc.variables['evratio'][time:end_time])

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during reading of %s' % (fname)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    if ( hybrid ):
        return truth, prior, posterior, obs, obs_operator, obs_err_var, prior_mean, posterior_mean, niters, evratio
    else:
        if   ( 'niters' in nc.variables.keys() ):
            return truth, prior, posterior, obs, obs_operator, obs_err_var, niters
        elif ( 'evratio' in nc.variables.keys() ):
            return truth, prior, posterior, obs, obs_operator, obs_err_var, evratio
        else:
            return truth, prior, posterior, obs, obs_operator, obs_err_var
# }}}
###############################################################
