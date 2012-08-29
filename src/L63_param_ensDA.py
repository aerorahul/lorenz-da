#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_param_ensDA.py - parameters for ensemble DA on L63
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

model      = type('',(),{})        # model Class
model.Name = 'L63'                 # model name
model.Ndof = 3                     # model degrees of freedom
model.Par  = [10.0, 28.0, 8.0/3.0] # model parameters
model.dt   = 1.0e-2                # model time-step

DA        = type('',(),{})      # data assimilation Class
DA.nassim = 200                 # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

Q = np.ones(model.Ndof)                # model error covariance ( covariance model is white for now )
Q = np.diag(Q) * 0.0

H = np.ones(model.Ndof)                # obs operator ( eye(Ndof) gives identity obs )
H = np.diag(H)

R = np.ones(model.Ndof)                # observation error covariance
R = 2.0 * R
R = np.diag(R)

ensDA              = type('',(),{})         # ensemble data assimilation Class
ensDA.inflation    = type('',(),{})         # inflation Class
ensDA.localization = type('',(),{})         # localization Class
ensDA.update                  = 2           # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 100         # number of ensemble members
ensDA.inflation.inflate       = 1           # inflation (0= None, 1= Multiplicative [1.01], 2= Additive [0.01],
                                            # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0])
ensDA.inflation.infl_fac      = 1.1         # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = 0           # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
ensDA.localization.cov_cutoff = 1.0         # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
ensDA.localization.cov_trunc  = model.Ndof  # truncate localization matrix (cov_trunc <= model.Ndof)

# name and attributes of/in the output diagnostic file
diag_file            = type('',(),{})  # diagnostic file Class
diag_file.filename   = model.Name + '_ensDA_diag.nc4'
diag_file.attributes = {'model'       : model.Name,
                        'sigma'       : model.Par[0],
                        'rho'         : model.Par[1],
                        'beta'        : model.Par[2],
                        'ntimes'      : DA.ntimes,
                        'dt'          : model.dt,
                        'Eupdate'     : ensDA.update,
                        'Elocalize'   : ensDA.localization.localize,
                        'Ecov_cutoff' : ensDA.localization.cov_cutoff,
                        'Ecov_trunc'  : ensDA.localization.cov_trunc,
                        'inflate'     : ensDA.inflation.inflate,
                        'infl_fac'    : ensDA.inflation.infl_fac}

# restart conditions
restart          = type('',(),{})    # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
