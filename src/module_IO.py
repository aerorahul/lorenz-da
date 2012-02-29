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
def create_diag(fname, ndof, nobs=None, nens=None):
# {{{
    '''
    create an output file for writing diagnostics

    create_diag(fname, ndof, nens=None, nobs=None)

    fname - name of the output file
     ndof - no. of degrees of freedom in the model
     nobs - no. of observations (None)
     nens - no. of ensemble members (None)
    '''

    if ( nobs == None ):
        nobs = ndof

    try:
        nc  = Dataset(fname, mode='w', clobber=True, format='NETCDF4')
        Dim = nc.createDimension('ntime',size=None)
        Dim = nc.createDimension('ndof', size=ndof)
        Dim = nc.createDimension('nobs', size=nobs)
        if not ( nens == None ):
            Dim = nc.createDimension('ncopy',size=nens)
        Var = nc.createVariable('truth',      'f8',('ntime','ndof',))
        if ( nens == None ):
            Var = nc.createVariable('prior',    'f8',('ntime','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','ndof',))
        else:
            Var = nc.createVariable('prior',    'f8',('ntime','ncopy','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','ncopy','ndof',))
        Var = nc.createVariable('obs',        'f8',('ntime','nobs',))
        Var = nc.createVariable('obs_err_var','f8',('ntime','nobs',))
        nc.close()
    except Exception as Instance:
        print 'Exception occured during creating ' + fname
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_diag(fname, time, truth, prior, posterior, obs, obs_err_var):
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
    obs_err_var - observation error variance
    '''

    if not os.path.isfile(fname):
        print 'file does not exist ' + fname
        sys.exit(2)

    try:
        nc = Dataset(fname, mode='a', clobber=True, format='NETCDF4')
        nc.variables['truth'][time,:]       = truth.copy()
        if ( len(np.shape(prior)) == 1 ):
            nc.variables['prior'][time,:]     = np.transpose(prior.copy())
            nc.variables['posterior'][time,:] = np.transpose(posterior.copy())
        else:
            nc.variables['prior'][time,:,:]     = np.transpose(prior.copy())
            nc.variables['posterior'][time,:,:] = np.transpose(posterior.copy())
        nc.variables['obs'][time,:]         = obs.copy()
        nc.variables['obs_err_var'][time,:] = obs_err_var.copy()
        nc.close()
    except Exception as Instance:
        print 'Exception occured during writing to ' + fname
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################
