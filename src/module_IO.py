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
import getopt
import numpy   as     np
from   netCDF4 import Dataset
###############################################################

###############################################################
module = 'module_IO.py'
###############################################################

###############################################################
def create_diag(dfile, ndof, nobs=None, nens=None, hybrid=False):
# {{{
    '''
    create an output file for writing diagnostics

    create_diag(dfile, ndof, nens=None, nobs=None)

    dfile - diagnostic file Class
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

        nc  = Dataset(dfile.filename, mode='w', clobber=True, format='NETCDF4')

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
        Var = nc.createVariable('obs_operator','f8',('ntime','ndof',))
        Var = nc.createVariable('obs_err_var', 'f8',('ntime','nobs',))

        if ( hybrid ):
            Var = nc.createVariable('central_prior',    'f8',('ntime','ndof',))
            Var = nc.createVariable('central_posterior','f8',('ntime','ndof',))
            Var = nc.createVariable('niters',           'f8',('ntime',))

        for (key,value) in dfile.attributes.iteritems():
            exec( 'nc.%s = value' % (key) )

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during creating  %s' % (dfile.filename)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_diag(fname, time, truth, prior, posterior, obs, obs_operator, obs_err_var, central_prior=None, central_posterior=None, niters=None, evratio=None):
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
      central_prior - central prior (None)
  central_posterior - central posterior (None)
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

        if not ( central_prior == None ):
            nc.variables['central_prior'][time,:] = central_prior.copy()

        if not ( central_posterior == None ):
            nc.variables['central_posterior'][time,:] = central_posterior.copy()

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
def read_diag_info(fname):
# {{{
    '''
    read the meta data from an output diagnostic file given name
    and reconstruct the classes for model, DA etc.

    read_diag_info(fname)

              fname - name of the file to read from, must already exist
              model - model class
                 DA - DA class
              ensDA - ensemble DA class
              varDA - variational DA class
    '''

    source = 'read_diag_info'

    if not os.path.isfile(fname):
        print 'error occured in %s of %s' % (source, module)
        print 'during the reading of %s' % (fname)
        print 'Error: File does not exist'
        sys.exit(2)

    model = type('',(),{})
    DA    = type('',(),{})
    ensDA = type('',(),{})
    varDA = type('',(),{})

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        model.Name = nc.model
        model.Ndof = len(nc.dimensions['ndof'])
        model.dt   = nc.dt
        if   ( model.Name == 'L63' ):
            model.Par = [nc.sigma, nc.rho, nc.beta]
        elif ( model.Name == 'L96' ):
            model.Par = [nc.F, nc.F+nc.dF]
        else:
            print 'model %s is not implemented' % (model.Name)
            sys.exit(2)

        DA.nassim = len(nc.dimensions['ntime'])
        DA.ntimes = nc.ntimes
        DA.Nobs   = len(nc.dimensions['nobs'])
        DA.t0     = 0.0

        if 'do_hybrid' in nc.ncattrs():
            DA.do_hybrid   = nc.do_hybrid
            DA.hybrid_wght = nc.hybrid_wght
            DA.hybrid_rcnt = nc.hybrid_rcnt
        else:
            DA.do_hybrid   = False

        if 'Eupdate' in nc.ncattrs():
            ensDA.update = nc.Eupdate
            ensDA.Nens   = len(nc.dimensions['ncopy'])
            ensDA.inflation         = type('', (), {})
            ensDA.inflation.infl_meth = nc.infl_meth
            ensDA.inflation.infl_fac  = nc.infl_fac
            ensDA.localization            = type('', (), {})
            ensDA.localization.localize   = nc.localize
            ensDA.localization.cov_cutoff = nc.cov_cutoff

        if 'Vupdate' in nc.ncattrs():
            varDA.update = nc.Vupdate
            varDA.minimization = type('', (), {})
            varDA.minimization.maxiter = nc.maxiter
            varDA.minimization.alpha   = nc.alpha
            varDA.minimization.cg      = nc.cg
            varDA.minimization.tol     = nc.tol

            if ( (varDA.update == 2) or (varDA.update == 4) ):
                varDA.fdvar           = type('',(),{})
                varDA.fdvar.offset    = nc.offset
                varDA.fdvar.window    = nc.window
                varDA.fdvar.nobstimes = nc.nobstimes
                varDA.fdvar.maxouter  = nc.maxouter

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during reading of %s' % (fname)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return [model, DA, ensDA, varDA]

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
      central_prior - central prior     ( if doing hybrid )
  central_posterior - central posterior ( if doing hybrid )
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
            hybrid_rcnt = nc.hybrid_rcnt
        else:
            hybrid = False

        if ( hybrid ):
            central_prior     = np.squeeze(nc.variables['central_prior'][time:end_time,])
            central_posterior = np.squeeze(nc.variables['central_posterior'][time:end_time,])

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
        return truth, prior, posterior, obs, obs_operator, obs_err_var, central_prior, central_posterior, niters, evratio
    else:
        if   ( 'niters' in nc.variables.keys() ):
            return truth, prior, posterior, obs, obs_operator, obs_err_var, niters
        elif ( 'evratio' in nc.variables.keys() ):
            return truth, prior, posterior, obs, obs_operator, obs_err_var, evratio
        else:
            return truth, prior, posterior, obs, obs_operator, obs_err_var
# }}}
###############################################################

###############################################################
def get_input_arguments():
# {{{
    '''
    get input arguments from command line

    [model, filename, start, end] = get_input_arguments()
    model    - name of the model []
    filename - file name to read []
    start    - starting index [-1]
    end      - ending index [-1]
    '''

    source = 'get_input_arguments'

    model    = []
    filename = []
    start    = -1
    end      = -1

    try:
        opts, args = getopt.getopt(sys.argv[1:],'m:f:s:e:h',['model=','filename=','start=','end=','help'])
    except Exception as Instance:
        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during reading arguments'
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    for a, o, in opts:
        if a in ('-h', '--help'):
            print 'no help has been written for %s in %s' % (source, module)
            print 'see code for details'
            sys.exit(0)
        elif a in ('-m','--model'):
            model = o
        elif a in ('-f','--filename'):
            filename = o
        elif a in ('-s','--start'):
            start = int(o)
        elif a in ('-e','--end'):
            end = int(o)
        else:
            assert False, 'unhandled option in %s of %s' % (source, module)
            sys.exit(0)

    returned_args = [model, filename, start, end]

    return returned_args

# }}}
###############################################################

###############################################################
def create_truth(tfile, ndof, nobs=None):
# {{{
    '''
    create a truth file for writing truth and observations

    create_truth(tfile, ndof, nobs=None)

    tfile - truth file Class
     ndof - number of degrees of freedom in the model
     nobs - number of observations (None)
    '''

    source = 'create_truth'

    if ( nobs == None ): nobs = ndof

    try:

        nc  = Dataset(tfile.filename, mode='w', clobber=True, format='NETCDF4')

        Dim = nc.createDimension('ntime',size=None)
        Dim = nc.createDimension('ndof', size=ndof)
        Dim = nc.createDimension('nobs', size=nobs)

        Var = nc.createVariable('truth',       'f8',('ntime','ndof',))
        Var = nc.createVariable('obs',         'f8',('ntime','ndof',))
        Var = nc.createVariable('obs_operator','f8',('ntime','nobs',))
        Var = nc.createVariable('obs_err_var', 'f8',('ntime','nobs',))

        for (key,value) in tfile.attributes.items():
            exec( 'nc.%s = %s' % (key,value) )

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during creating  %s' % (tfile.filename)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_truth(tname, time, truth, obs, obs_operator, obs_err_var):
# {{{
    '''
    write the truth and observations to the output file

    write_truth(tname, time, truth, obs, obs_operator, obs_err_var)

              tname - name of the output file, must already exist
               time - time index to write diagnostics for
              truth - truth
                obs - observations
       obs_operator - forward observation operator
        obs_err_var - observation error variance
    '''

    source = 'write_truth'

    if not os.path.isfile(tname):
        print 'file does not exist ' + tname
        sys.exit(2)

    try:

        nc = Dataset(tname, mode='a', clobber=True, format='NETCDF4')

        nc.variables['truth'][       time,:] =        truth.copy()
        nc.variables['obs'][         time,:] =          obs.copy()
        nc.variables['obs_operator'][time,:] = obs_operator.copy()
        nc.variables['obs_err_var'][ time,:] =  obs_err_var.copy()

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
def read_truth(fname, time, end_time=None):
# {{{
    '''
    read the truth and observations from an output file given name and time index

    read_truth(fname, time, end_time=None)

              fname - name of the file to read from, must already exist
               time - time index to read diagnostics
           end_time - return chunk of data from time to end_time (None)
              truth - truth
                obs - observations
       obs_operator - forward observation operator
        obs_err_var - observation error variance
    '''

    source = 'read_truth'

    if not os.path.isfile(fname):
        print 'file does not exist ' + fname
        sys.exit(2)

    if ( end_time == None ): end_time = time + 1

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        truth        = np.squeeze(nc.variables['truth'][time:end_time,])
        obs          = np.squeeze(nc.variables['obs'][time:end_time,])
        obs_operator = np.squeeze(nc.variables['obs_operator'][time:end_time,])
        tmp          = np.squeeze(nc.variables['obs_err_var'][time:end_time,])

        if ( end_time - time == 1 ):
            obs_err_var = np.diag(tmp)
        else:
            obs_err_var = np.zeros((np.shape(tmp)[0],np.shape(tmp)[1],np.shape(tmp)[1]))
            for k in range(0, np.shape(tmp)[0]):
                obs_err_var[k,] = np.diag(tmp[k,])

        nc.close()

    except Exception as Instance:

        print 'Exception occured in %s of %s' % (source, module)
        print 'Exception occured during reading of %s' % (fname)
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    return truth, obs, obs_operator, obs_err_var
# }}}
###############################################################
