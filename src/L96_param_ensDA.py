#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_param_ensDA.py - parameters for ensemble DA on L96
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

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 8.4]         # model parameters F, dF
model.dt   = 1.0e-4             # model time-step

DA        = type('', (), {})    # data assimilation Class
DA.nassim = 2000                # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time
DA.Nobs   = 20                  # no. of obs to assimilate ( DA.Nobs <= model.Ndof)

Q = np.ones(model.Ndof)                   # model error covariance ( covariance model is white for now )
Q = np.diag(Q) * 0.0

H = np.ones(model.Ndof)                   # obs operator ( eye(Ndof) gives identity obs )
if ( DA.Nobs != model.Ndof ):
    index = np.arange(model.Ndof)
    np.random.shuffle(index)
    H[index[:-DA.Nobs]] = np.NaN
H = np.diag(H)

R = np.ones(model.Ndof)                   # observation error covariance
R = R + np.random.rand(model.Ndof)
R = np.diag(R)

ensDA              = type('', (), {})  # ensemble data assimilation Class
ensDA.inflation    = type('', (), {})  # inflation Class
ensDA.localization = type('', (), {})  # localization Class
ensDA.update                  = 2      # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 30     # number of ensemble members
ensDA.inflation.infl_meth     = 1      # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                       # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
ensDA.inflation.infl_fac      = 1.06   # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = True   # do localization
ensDA.localization.cov_cutoff = 0.0625 # normalized covariance cutoff = cutoff / ( 2*normalized_dist)

# name and attributes of/in the output diagnostic file
diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = model.Name + '_ensDA_diag.nc4'
diag_file.attributes = {'model'       : model.Name,
                        'F'           : model.Par[0],
                        'dF'          : model.Par[1]-model.Par[0],
                        'ntimes'      : DA.ntimes,
                        'dt'          : model.dt,
                        'Eupdate'     : ensDA.update,
                        'localize'    : int(ensDA.localization.localize),
                        'cov_cutoff'  : ensDA.localization.cov_cutoff,
                        'infl_meth'   : ensDA.inflation.infl_meth,
                        'infl_fac'    : ensDA.inflation.infl_fac}

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
