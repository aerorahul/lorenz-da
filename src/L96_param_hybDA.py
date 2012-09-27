#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_param_hybDA.py - parameters for hybrid DA on L96
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

model      = type('',(),{})     # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 8.4]         # model parameters
model.dt   = 1.0e-4             # model time-step

DA             = type('',(),{}) # data assimilation Class
DA.nassim      = 1000           # no. of assimilation cycles
DA.ntimes      = 0.1            # do assimilation every ntimes non-dimensional time units
DA.t0          = 0.0            # initial time
DA.do_hybrid   = True           # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
DA.hybrid_rcnt = True           # True= re-center ensemble about varDA, False= free ensDA
DA.hybrid_wght = 1.0            # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)

Q = np.ones(model.Ndof)                   # model error covariance ( covariance model is white for now )
Q = np.diag(Q) * 0.0

H = np.ones(model.Ndof)                   # obs operator ( eye(Ndof) gives identity obs )
H[::2] = np.NaN
H = np.diag(H)

R = np.ones(model.Ndof)                   # observation error covariance
R[1::2] = np.sqrt(2.0)
R[1::4] = np.sqrt(3.0)
R = np.diag(R)

ensDA              = type('',(),{})            # ensemble data assimilation Class
ensDA.inflation    = type('',(),{})            # inflation Class
ensDA.localization = type('',(),{})            # localization Class
ensDA.update                  = 2              # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 30             # number of ensemble members
ensDA.inflation.inflate       = 1              # inflation (0= None, 1= Multiplicative [1.01], 2= Additive [0.01],
                                               # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0])
ensDA.inflation.infl_fac      = 1.06           # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = 1              # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
ensDA.localization.cov_cutoff = 0.0625         # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
ensDA.localization.cov_trunc  = model.Ndof     # truncate localization matrix (cov_trunc <= model.Ndof)

varDA                         = type('',(),{}) # variational data assimilation Class
varDA.minimization            = type('',(),{}) # minimization Class
varDA.inflation               = type('',(),{}) # inflation Class
varDA.localization            = type('',(),{}) # localization Class
varDA.update                  = 2              # variational-based DA method (1= 3Dvar; 2= 4Dvar)
varDA.precondition            = True           # precondition before minimization
varDA.maxouter                = 1              # no. of outer loops
varDA.minimization.maxiter    = 1000           # maximum iterations for minimization
varDA.minimization.tol        = 1e-4           # tolerance to end the variational minimization iteration
varDA.inflation.inflate       = True           # inflate [ > 1.0 ] / deflate [ < 1.0 ] static covariance
varDA.inflation.infl_fac      = 2.75           # inflate static covariance
varDA.localization.localize   = 1              # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
varDA.localization.cov_cutoff = 0.0625         # normalized covariance cutoff = cutoff / ( 2*normalized_dist )
varDA.localization.cov_trunc  = model.Ndof     # truncate localization matrix (cov_trunc <= model.Ndof)

if ( varDA.update == 2 ):
    varDA.fdvar                = type('',(),{}) # 4DVar Class
    varDA.fdvar.window         = 0.075          # length of the 4Dvar assimilation window
    varDA.fdvar.offset         = 0.25           # time offset: forecast from analysis to background time
    varDA.fdvar.nobstimes      = 4              # no. of evenly spaced obs. times in the window

# name and attributes of/in the output diagnostic file
diag_file            = type('',(),{})  # diagnostic file Class
diag_file.filename   = model.Name + '_hybDA_diag.nc4'
diag_file.attributes = {'model'       : model.Name,
                        'F'           : model.Par[0],
                        'dF'          : model.Par[1]-model.Par[0],
                        'dt'          : model.dt,
                        'ntimes'      : DA.ntimes,
                        'do_hybrid'   : int(DA.do_hybrid),
                        'hybrid_rcnt' : int(DA.hybrid_rcnt),
                        'hybrid_wght' : DA.hybrid_wght,
                        'Eupdate'     : ensDA.update,
                        'Elocalize'   : ensDA.localization.localize,
                        'Ecov_cutoff' : ensDA.localization.cov_cutoff,
                        'Ecov_trunc'  : ensDA.localization.cov_trunc,
                        'Einflate'    : ensDA.inflation.inflate,
                        'Einfl_fac'   : ensDA.inflation.infl_fac,
                        'Vupdate'     : varDA.update,
                        'Vinflate'    : int(varDA.inflation.inflate),
                        'Vinfl_fac'   : varDA.inflation.infl_fac,
                        'Vlocalize'   : varDA.localization.localize,
                        'Vcov_cutoff' : varDA.localization.cov_cutoff,
                        'Vcov_trunc'  : varDA.localization.cov_trunc,
                        'precondition': int(varDA.precondition),
                        'maxouter'    : varDA.maxouter,
                        'maxiter'     : varDA.minimization.maxiter,
                        'tol'         : varDA.minimization.tol}
if ( varDA.update == 2 ):
    diag_file.attributes.update({'offset'    : varDA.fdvar.offset,
                                 'window'    : varDA.fdvar.window,
                                 'nobstimes' : varDA.fdvar.nobstimes})

# restart conditions
restart          = type('',(),{})    # restart initial conditions Class
restart.time     = -1                # None == default | -N...-1 0 1...N
restart.filename = '/home/rmahajan/svn-work/lorenz1963/data/L96/ensDA_N=30/inf=1.06/L96_ensDA_diag-0.nc4'
