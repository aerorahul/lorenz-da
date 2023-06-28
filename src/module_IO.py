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
from module_Lorenz import Lorenz
from module_DA import DataAssim, EnsDataAssim, VarDataAssim
###############################################################

###############################################################
module = 'module_IO.py'
###############################################################

###############################################################
def create_diag(dfile, ndof, nouter=1, nobs=None, nens=None, hybrid=False):
# {{{
    '''
    create an output file for writing diagnostics

    create_diag(dfile, ndof, nens=None, nobs=None)

    dfile - diagnostic file Class
     ndof - number of degrees of freedom in the model
   nouter - number of outer loops (1)
     nobs - number of observations (None)
     nens - number of ensemble members (None)
   hybrid - flag for hybrid DA (False)
    '''

    source = 'create_diag'

    if ( nobs is None ):
        nobs = ndof

    if ( ( hybrid ) and ( nens is None ) ):
        print('nens cannot be None if doing hybrid')
        sys.exit(2)

    try:

        nc  = Dataset(dfile.filename, mode='w', clobber=True, format='NETCDF4')

        Dim = nc.createDimension('ntime', size=None)
        Dim = nc.createDimension('nouter',size=nouter)
        Dim = nc.createDimension('ndof',  size=ndof)
        Dim = nc.createDimension('nobs',  size=nobs)

        if not ( nens is None ):
            Dim = nc.createDimension('ncopy',size=nens)

        Var = nc.createVariable('truth','f8',('ntime','ndof',))

        if ( nens is None ):
            Var = nc.createVariable('prior',    'f8',('ntime','nouter','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','nouter','ndof',))
            Var = nc.createVariable('niters',   'f8',('ntime','nouter',))
        else:
            Var = nc.createVariable('prior',    'f8',('ntime','nouter','ncopy','ndof',))
            Var = nc.createVariable('posterior','f8',('ntime','nouter','ncopy','ndof',))
            Var = nc.createVariable('evratio',  'f8',('ntime','nouter',))

        Var = nc.createVariable('obs',         'f8',('ntime','nobs',))
        Var = nc.createVariable('obs_operator','f8',('ntime','ndof',))
        Var = nc.createVariable('obs_err_var', 'f8',('ntime','ndof',))

        if ( hybrid ):
            Var = nc.createVariable('central_prior',    'f8',('ntime','nouter','ndof',))
            Var = nc.createVariable('central_posterior','f8',('ntime','nouter','ndof',))
            Var = nc.createVariable('niters',           'f8',('ntime','nouter',))

        for (key,value) in dfile.attributes.items():
            exec( 'nc.%s = value' % (key) )

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during creating  %s' % (dfile.filename))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_diag(fname, time, outer, truth, prior, posterior, obs, obs_operator, obs_err_var,
        central_prior=None, central_posterior=None, niters=None, evratio=None):
# {{{
    '''
    write the diagnostics to an output file

    write_diag(fname, time, truth, prior, posterior, obs, obs_err_var)

              fname - name of the output file, must already exist
               time - time index to write diagnostics for
              outer - outer loop index to write diagnostics for
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
        print('file does not exist ' + fname)
        sys.exit(2)

    try:

        nc = Dataset(fname, mode='a', clobber=True, format='NETCDF4')

        nc.variables['prior'    ][time,outer,:] = prior.copy()
        nc.variables['posterior'][time,outer,:] = posterior.copy()

        if ( outer == 0 ):
            nc.variables['truth'       ][time,:] = truth.copy()
            nc.variables['obs'         ][time,:] = obs.copy()
            nc.variables['obs_operator'][time,:] = obs_operator.copy()
            nc.variables['obs_err_var' ][time,:] = obs_err_var.copy()

        if not ( central_prior is None ):
            nc.variables['central_prior'    ][time,outer,:] = central_prior.copy()

        if not ( central_posterior is None ):
            nc.variables['central_posterior'][time,outer,:] = central_posterior.copy()

        if not ( niters is None ):
            nc.variables['niters'][time,outer] = niters

        if not ( evratio is None ):
            nc.variables['evratio'][time,outer] = evratio

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during writing to %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
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
        print('error occured in %s of %s' % (source, module))
        print('during the reading of %s' % (fname))
        print('Error: File does not exist')
        sys.exit(2)

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        Name = nc.model
        Ndof = len(nc.dimensions['ndof'])
        dt   = nc.dt
        if   ( Name == 'L63' ):
            Par = [nc.sigma, nc.rho, nc.beta]
        elif ( Name == 'L96' ):
            Par = [nc.F, nc.F+nc.dF]
        else:
            print('model %s is not implemented' % (Name))
            sys.exit(2)

        if ( Name in ['L63', 'L96'] ):
            model = Lorenz()
            model.init(Name=Name,Ndof=Ndof,Par=Par,dt=dt)

        nassim   = len(nc.dimensions['ntime'])
        ntimes   = nc.ntimes
        Nobs     = len(nc.dimensions['nobs'  ]) if ( 'nobs'   in nc.dimensions ) else model.Ndof
        maxouter = len(nc.dimensions['nouter']) if ( 'nouter' in nc.dimensions ) else 1

        DA = DataAssim()
        DA.init(nassim=nassim,ntimes=ntimes,maxouter=maxouter,Nobs=Nobs)

        if 'do_hybrid' in nc.ncattrs():
            setattr(DA,'do_hybrid',  nc.do_hybrid)
            setattr(DA,'hybrid_wght',nc.hybrid_wght)
            setattr(DA,'hybrid_rcnt',nc.hybrid_rcnt)
        else:
            setattr(DA,'do_hybrid',  False)

        ensDA = EnsDataAssim()
        if 'Eupdate' in nc.ncattrs():
            update     = nc.Eupdate
            Nens       = len(nc.dimensions['ncopy'])
            inflate    = nc.Einflate
            infl_fac   = nc.Einfl_fac
            localize   = nc.Elocalize
            cov_cutoff = nc.Ecov_cutoff
            cov_trunc  = nc.Ecov_trunc
            ensDA.init(model,DA,\
                       update=update,Nens=Nens,\
                       inflate=inflate,infl_fac=infl_fac,\
                       localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc)

        varDA = VarDataAssim()
        if 'Vupdate' in nc.ncattrs():
            update       = nc.Vupdate
            precondition = nc.precondition
            maxiter      = nc.maxiter
            tol          = nc.tol
            inflate      = nc.Vinflate
            infl_fac     = nc.Vinfl_fac
            infl_adp     = nc.Vinfl_adp
            localize     = nc.Vlocalize
            cov_cutoff   = nc.Vcov_cutoff
            cov_trunc    = nc.Vcov_trunc
            window       = nc.window
            offset       = nc.offset
            nobstimes    = nc.nobstimes
            varDA.init(model,DA,\
                       update=update,precondition=precondition,\
                       maxiter=maxiter,tol=tol,\
                       inflate=inflate,infl_fac=infl_fac,infl_adp=infl_adp,\
                       localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc,\
                       window=window,offset=offset,nobstimes=nobstimes)

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading of %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
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
        print('file does not exist ' + fname)
        sys.exit(2)

    if ( end_time is None ): end_time = time + 1

    [model, DA, ensDA, varDA] = read_diag_info(fname)

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        truth        = np.squeeze(nc.variables[ 'truth'       ][time:end_time,])
        prior        = np.squeeze(nc.variables[ 'prior'       ][time:end_time,])
        posterior    = np.squeeze(nc.variables[ 'posterior'   ][time:end_time,])
        obs          = np.squeeze(nc.variables[ 'obs'         ][time:end_time,])
        obs_operator = np.squeeze(nc.variables[ 'obs_operator'][time:end_time,])
        obs_err_var  = np.squeeze(nc.variables['obs_err_var' ][time:end_time,])

        if ( DA.do_hybrid ):
            central_prior     = np.squeeze(nc.variables['central_prior'    ][time:end_time,])
            central_posterior = np.squeeze(nc.variables['central_posterior'][time:end_time,])

        if 'niters' in list(nc.variables.keys()):
            niters = nc.variables['niters'][time:end_time]

        if 'evratio' in list(nc.variables.keys()):
            evratio = nc.variables['evratio'][time:end_time]

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading of %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    if ( DA.do_hybrid ):
        return truth, prior, posterior, obs, obs_operator, obs_err_var, central_prior, central_posterior, niters, evratio
    else:
        if   ( 'niters' in list(nc.variables.keys()) ):
            return truth, prior, posterior, obs, obs_operator, obs_err_var, niters
        elif ( 'evratio' in list(nc.variables.keys()) ):
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
        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading arguments')
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    for a, o, in opts:
        if a in ('-h', '--help'):
            print('no help has been written for %s in %s' % (source, module))
            print('see code for details')
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

    if ( nobs is None ): nobs = ndof

    try:

        nc  = Dataset(tfile.filename, mode='w', clobber=True, format='NETCDF4')

        Dim = nc.createDimension('ntime',size=None)
        Dim = nc.createDimension('ndof', size=ndof)
        Dim = nc.createDimension('nobs', size=nobs)

        Var = nc.createVariable('truth',       'f8',('ntime','ndof',))
        Var = nc.createVariable('obs',         'f8',('ntime','ndof',))
        Var = nc.createVariable('obs_operator','f8',('ntime','nobs',))
        Var = nc.createVariable('obs_err_var', 'f8',('ntime','nobs',))

        for (key,value) in list(tfile.attributes.items()):
            exec( 'nc.%s = value' % (key) )

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during creating  %s' % (tfile.filename))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
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
        print('file does not exist ' + tname)
        sys.exit(2)

    try:

        nc = Dataset(tname, mode='a', clobber=True, format='NETCDF4')

        nc.variables['truth'][       time,:] =        truth.copy()
        nc.variables['obs'][         time,:] =          obs.copy()
        nc.variables['obs_operator'][time,:] = obs_operator.copy()
        nc.variables['obs_err_var'][ time,:] =  obs_err_var.copy()

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during writing to %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
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
        print('file does not exist ' + fname)
        sys.exit(2)

    if ( end_time is None ): end_time = time + 1

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

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading of %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    return truth, obs, obs_operator, obs_err_var
# }}}
###############################################################

###############################################################
def transfer_ga(file_src, file_dst):
# {{{
    '''
    transfer global attributes from one netCDF file to another

    transfer_ga(file_src, file_dst)

    file_src - global attributes to read from ( file_src must exist )
    file_dst - global attributes to write to  ( file_dst must exist )
    '''

    source = 'transfer_ga'

    if not os.path.isfile(file_src):
        print('file to read global attributes from does not exist ' + file_src)
        sys.exit(2)

    if not os.path.isfile(file_dst):
        print('file to write global attributes to does not exist ' + file_dst)
        sys.exit(2)

    try:

        nc_src = Dataset(file_src, mode='r', clobber=False, format='NETCDF4')
        nc_dst = Dataset(file_dst, mode='a', clobber=False, format='NETCDF4')

        for attr_name in nc_src.ncattrs():
            exec( 'attr_value = nc_src.%s' % (attr_name) )
            exec( 'nc_dst.%s = attr_value' % (attr_name) )

        nc_src.close()
        nc_dst.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during transfering global attributes from %s to %s' % ( file_src, file_dst ))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def create_ObImpact_diag(fname, model, DA, ensDA, varDA, generic=False):
# {{{
    '''
    create an output file for writing observation impact diagnostics

    create_ObImpact_diag(dfile, model)

    fname - name of the observation impact file to create
    model - model Class
       DA - DA Class
    ensDA - ensemble DA Class
    varDA - variational DA Class
  generic - generic ObImpact diagnostic file [default=False]
    '''

    source = 'create_ObImpact_diag'

    try:

        nc  = Dataset(fname, mode='w', clobber=True, format='NETCDF4')

        Dim = nc.createDimension('ntime', size=None       )
        Dim = nc.createDimension('ndof',  size=model.Ndof )
        Dim = nc.createDimension('nobs',  size=DA.Nobs    )
        Dim = nc.createDimension('nouter',size=DA.maxouter)

        if ( generic ):
            Var = nc.createVariable('dJa','f8',('ntime','nobs',))
            Var = nc.createVariable('dJb','f8',('ntime','nobs',))
        else:
            Var = nc.createVariable('ens_dJa','f8',('ntime','nobs',))
            Var = nc.createVariable('ens_dJb','f8',('ntime','nobs',))
            Var = nc.createVariable('adj_dJa','f8',('ntime','nobs',))
            Var = nc.createVariable('adj_dJb','f8',('ntime','nobs',))

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during creating %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def write_ObImpact_diag(fname, time, dJa=None, dJb=None, ens_dJa=None, ens_dJb=None, adj_dJa=None, adj_dJb=None):
# {{{
    '''
    write the observation impact diagnostics to an output file

    write_ObImpact_diag(fname, time, dJa=None, dJb=None, ens_dJa=None, ens_dJb=None, adj_dJa=None, adj_dJb=None)

      fname - name of the output file, must already exist
       time - time index to write diagnostics for
        dJa - 2nd order correction to the estimate of observation impact
        dJb - 1st order estimate of observation impact
    ens_dJa - 2nd order correction to ensemble estimate of observation impact
    ens_dJb - 1st order ensemble estimate of observation impact
    adj_dJa - 2nd order correction to adjoint estimate of observation impact
    adj_dJb - 1st order adjoint estimate of observation impact
    '''

    source = 'write_ObImpact_diag'

    if not os.path.isfile(fname):
        print('file does not exist ' + fname)
        sys.exit(2)

    try:

        nc = Dataset(fname, mode='a', clobber=True, format='NETCDF4')

        if (     dJa is not None ): nc.variables[    'dJa'][time,:] =     dJa
        if (     dJb is not None ): nc.variables[    'dJb'][time,:] =     dJb
        if ( ens_dJa is not None ): nc.variables['ens_dJa'][time,:] = ens_dJa
        if ( ens_dJb is not None ): nc.variables['ens_dJb'][time,:] = ens_dJb
        if ( adj_dJa is not None ): nc.variables['adj_dJa'][time,:] = adj_dJa
        if ( adj_dJb is not None ): nc.variables['adj_dJb'][time,:] = adj_dJb

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during writing to %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    return
# }}}
###############################################################

###############################################################
def read_ObImpact_diag(fname, time, end_time=None, generic=False):
# {{{
    '''
    read the observation impact diagnostics from an output file given name and time index

    read_ObImpact_diag(fname, time, end_time=None)

       fname - name of the file to read from, must already exist
        time - time index to read diagnostics
    end_time - return chunk of data from time to end_time (None)
     generic - generic ObImpact diagnostic file [default=False]
         dJa - 2nd order correction to the estimate of observation impact
         dJb - 1st order estimate of observation impact
     ens_dJa - 2nd order correction to ensemble estimate of observation impact
     ens_dJb - 1st order ensemble estimate of observation impact
     adj_dJa - 2nd order correction to adjoint estimate of observation impact
     adj_dJb - 1st order adjoint estimate of observation impact
    '''

    source = 'read_ObImpact_diag'

    if not os.path.isfile(fname):
        print('file does not exist ' + fname)
        sys.exit(2)

    if ( end_time is None ): end_time = time + 1

    try:

        nc = Dataset(fname, mode='r', format='NETCDF4')

        if ( generic ):
            dJa = nc.variables['dJa'][time:end_time,]
            dJb = nc.variables['dJb'][time:end_time,]
        else:
            ens_dJa = nc.variables['ens_dJa'][time:end_time,]
            ens_dJb = nc.variables['ens_dJb'][time:end_time,]
            adj_dJa = nc.variables['adj_dJa'][time:end_time,]
            adj_dJb = nc.variables['adj_dJb'][time:end_time,]

        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading of %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    if ( generic ):
        return [dJa, dJb]
    else:
        return [ens_dJa, ens_dJb, adj_dJa, adj_dJb]
# }}}
###############################################################

###############################################################
def read_clim_cov(model=None,fname=None,norm=False):
# {{{
    '''
    read the climatological covariance from file

    Bc = read_clim_cov(model=None,fname=None,norm=False)

    model - model Class [None]
    fname - filename to read climatological covariance from [None]
     norm - return normalized covariance [False]
       Bc - climatological covariance
    '''

    source = 'read_clim_cov'

    if ( model is None and fname is None ):
        print('Exception occured in %s of %s' % (source, module))
        print('must pass either model class or filename')
        sys.exit(0)

    if ( fname is None ): fname = '%s_climo_B.nc4' % model.Name

    print('load climatological covariance for from %s ...' % (fname))

    try:

        nc = Dataset(fname,'r')
        Bc = nc.variables['B'][:]
        nc.close()

    except Exception as Instance:

        print('Exception occured in %s of %s' % (source, module))
        print('Exception occured during reading of %s' % (fname))
        print(type(Instance))
        print(Instance.args)
        print(Instance)
        sys.exit(1)

    if ( norm ): Bc = Bc / np.max(np.diag(Bc))

    return Bc
# }}}
###############################################################

###############################################################
class Container(object):

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

    def __init__(self,**kwargs):
    #{{{
        '''
        Initializes a blank Container class for the purposes of
        writing a diagnostic file attributes, or
        restart file attributes
        '''

        for key, value in kwargs.items(): self.__setattr__(key,value)

        pass
    #}}}
###############################################################
