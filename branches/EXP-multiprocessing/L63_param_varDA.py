#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_param_varDA.py - parameters for variational DA on L63
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import numpy as np
###############################################################

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model      = type('',(),{})             # model Class
model.Name = 'L63'                      # model name
model.Ndof = 3                          # model degrees of freedom
model.Par  = [10.0, 28.0, 8.0/3.0]      # model parameters F, dF
model.dt   = 1.0e-2                     # model time-step

DA        = type('',(),{})      # DA class
DA.nassim = 200                 # no. of assimilation cycles
DA.ntimes = 0.25                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

Q = np.ones(model.Ndof)         # model error covariance ( covariance model is white for now )
Q = np.diag(Q) * 0.0

H = np.ones(model.Ndof)         # obs operator ( eye(Ndof) gives identity obs )
H = np.diag(H)

R = np.ones(model.Ndof)         # observation error covariance
R = 2 * R
R = np.diag(R)

varDA                         = type('',(),{}) # VarDA class
varDA.minimization            = type('',(),{}) # minimization class
varDA.localization            = type('',(),{}) # localization class
varDA.update                  = 1              # DA method (1= 3Dvar; 2= 4Dvar)
varDA.precondition            = 1              # precondition before minimization (0= None; 1= sqrtB; 2= FullB)
varDA.maxouter                = 1              # no. of outer loops
varDA.minimization.maxiter    = 1000           # maximum iterations
varDA.minimization.tol        = 1e-4           # tolerance to end the variational minimization iteration
varDA.localization.localize   = 1              # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
varDA.localization.cov_cutoff = 0.0625         # normalized covariance cutoff = cutoff / ( 2*normalized_dist )
varDA.localization.cov_trunc  = model.Ndof     # truncate localization matrix (cov_trunc <= model.Ndof)

if ( varDA.update == 2 ):
    varDA.fdvar                = type('',(),{}) # 4DVar class
    varDA.fdvar.window         = DA.ntimes      # length of the 4Dvar assimilation window
    varDA.fdvar.offset         = 0.5            # time offset: forecast from analysis to background time
    varDA.fdvar.nobstimes      = 5              # no. of evenly spaced obs. times in the window

diag_file            = type('',(),{})  # diagnostic file Class
diag_file.filename   = model.Name + '_varDA_diag.nc4'
diag_file.attributes = {'model'       : model.Name,
                        'sigma'       : model.Par[0],
                        'rho'         : model.Par[1],
                        'beta'        : model.Par[2],
                        'ntimes'      : DA.ntimes,
                        'dt'          : model.dt,
                        'Vupdate'     : varDA.update,
                        'maxouter'    : varDA.maxiter,
                        'precondition': varDA.precondition,
                        'Vlocalize'   : varDA.localization.localize,
                        'Vcov_cutoff' : varDA.localization.cov_cutoff,
                        'Vcov_trunc'  : varDA.localization.cov_trunc,
                        'maxiter'     : varDA.minimization.maxiter,
                        'tol'         : varDA.minimization.tol}
if ( varDA.update == 2 ):
    diag_file.attributes.update({'offset'    : varDA.fdvar.offset,
                                 'window'    : varDA.fdvar.window,
                                 'nobstimes' : varDA.fdvar.nobstimes})

# restart conditions
restart          = type('',(),{})  # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
