#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_param_hybDA.py - parameters for hybrid DA
###############################################################

###############################################################
import numpy as np
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################
###############################################################

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model = type('', (), {})         # model Class
model.Name = 'L63'                  # model name
model.Ndof = 3                      # model degrees of freedom
model.Par = [10.0, 28.0, 8.0 / 3.0]  # model parameters
model.dt = 1.0e-2                 # model time-step

DA = type('', (), {})   # data assimilation Class
DA.nassim = 200              # no. of assimilation cycles
DA.ntimes = 0.25             # do assimilation every ntimes non-dimensional time units
DA.t0 = 0.0              # initial time
# True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
DA.do_hybrid = True
# weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
DA.hybrid_wght = 0.5
# True= re-center ensemble about varDA, False= free ensDA
DA.hybrid_rcnt = True

# model error covariance ( covariance model is white for now )
Q = np.ones(model.Ndof)
Q = np.diag(Q) * 0.0

# obs operator ( eye(Ndof) gives identity obs )
H = np.ones(model.Ndof)
H = np.diag(H)

R = np.ones(model.Ndof)                   # observation error covariance
R = 2.0 * R
R = np.diag(R)

ensDA = type('', (), {})           # ensemble data assimilation Class
ensDA.inflation = type('', (), {})           # inflation Class
ensDA.localization = type('', (), {})           # localization Class
# ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.update = 2
ensDA.Nens = 100           # number of ensemble members
# inflate initial ensemble by init_ens_infl_fac
ensDA.init_ens_infl_fac = 1.0
# inflation (0= None, 1= Multiplicative [1.01], 2= Additive [0.01],
ensDA.inflation.inflate = 1
# 3= Cov. Relax [0.25], 4= Spread Restoration [1.0])
# Depends on inflation method (see values in [] above)
ensDA.inflation.infl_fac = 1.1
# localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
ensDA.localization.localize = 0
# normalized covariance cutoff = cutoff / ( 2*normalized_dist)
ensDA.localization.cov_cutoff = 1.0
# truncate localization matrix (cov_trunc <= model.Ndof)
ensDA.localization.cov_trunc = model.Ndof

varDA = type('', (), {})   # variational data assimilation Class
varDA.minimization = type('', (), {})   # minimization Class
varDA.localization = type('', (), {})   # localization Class
# variational-based DA method (1 = 3Dvar; 2= 4Dvar)
varDA.update = 1
# precondition before minimization (0= None; 1= sqrtB; 2= FullB)
varDA.precondition = 1
varDA.maxouter = 1                # no. of outer loops
# maximum iterations for minimization
varDA.minimization.maxiter = 1000
# tolerance to end the variational minimization iteration
varDA.minimization.tol = 1e-4
# localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
varDA.localization.localize = 0
# normalized covariance cutoff = cutoff / ( 2*normalized_dist)
varDA.localization.cov_cutoff = 1.0
# truncate localization matrix (cov_trunc <= model.Ndof)
varDA.localization.cov_trunc = model.Ndof

if (varDA.update == 2):
    varDA.fdvar = type('', (), {})  # 4DVar class
    varDA.fdvar.window = 0.025          # length of the 4Dvar assimilation window
    # time offset: forecast from analysis to background time
    varDA.fdvar.offset = 0.5
    # no. of evenly spaced obs. times in the window
    varDA.fdvar.nobstimes = 2

# name and attributes of/in the output diagnostic file
diag_file = type('', (), {})  # diagnostic file Class
diag_file.filename = model.Name + '_hybDA_diag.nc4'
diag_file.attributes = {'model': model.Name,
                        'sigma': model.Par[0],
                        'rho': model.Par[1],
                        'beta': model.Par[2],
                        'dt': model.dt,
                        'ntimes': DA.ntimes,
                        'do_hybrid': int(DA.do_hybrid),
                        'hybrid_wght': DA.hybrid_wght,
                        'hybrid_rcnt': int(DA.hybrid_rcnt),
                        'Eupdate': ensDA.update,
                        'Elocalize': ensDA.localization.localize,
                        'Ecov_cutoff': ensDA.localization.cov_cutoff,
                        'Ecov_trunc': ensDA.localization.cov_trunc,
                        'inflate': ensDA.inflation.inflate,
                        'infl_fac': ensDA.inflation.infl_fac,
                        'Vupdate': varDA.update,
                        'precondition': varDA.precondition,
                        'maxouter': varDA.maxouter,
                        'Vlocalize': varDA.localization.localize,
                        'Vcov_cutoff': varDA.localization.cov_cutoff,
                        'Vcov_trunc': varDA.localization.cov_trunc,
                        'maxiter': varDA.minimization.maxiter,
                        'tol': varDA.minimization.tol}
if (varDA.update == 2):
    diag_file.attributes.update({'offset': varDA.fdvar.offset,
                                 'window': varDA.fdvar.window,
                                 'nobstimes': varDA.fdvar.nobstimes})

# restart conditions
restart = type('', (), {})    # restart initial conditions Class
restart.time = None              # None == default | -N...-1 0 1...N
restart.filename = ''
