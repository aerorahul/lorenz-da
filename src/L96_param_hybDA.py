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

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 8.4]         # model parameters
model.dt   = 1.0e-4             # model time-step

DA             = type('', (), {}) # data assimilation Class
DA.nassim      = 465              # no. of assimilation cycles
DA.ntimes      = 0.05             # do assimilation every ntimes non-dimensional time units
DA.t0          = 0.0              # initial time
DA.Nobs        = 20               # no. of obs to assimilate ( DA.Nobs <= model.Ndof)
DA.do_hybrid   = True             # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
DA.hybrid_wght = 0.99             # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
DA.hybrid_rcnt = True             # True= re-center ensemble about varDA, False= free ensDA

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
ensDA.update                  = 2      # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 30     # number of ensemble members
ensDA.inflation.infl_meth     = 1      # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                       # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
ensDA.inflation.infl_fac      = 1.06   # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = True   # do localization
ensDA.localization.cov_cutoff = 0.0625 # normalized covariance cutoff = cutoff / ( 2*normalized_dist)

varDA                      = type('', (), {}) # variational data assimilation Class
varDA.minimization         = type('', (), {}) # minimization Class
varDA.update               = 1                # variational-based DA method (1 = 3Dvar; 2= 4Dvar)
varDA.minimization.maxiter = 1000             # maximum iterations for minimization
varDA.minimization.alpha   = 4e-4             # size of step in direction of normalized J
varDA.minimization.cg      = True             # True = Use conjugate gradient; False = Perform line search
varDA.minimization.tol     = 1e-3             # tolerance to end the variational minimization iteration

if ( (varDA.update == 2) or (varDA.update == 4) ): fdvar = True
else:                                              fdvar = False

if ( fdvar ):
    varDA.fdvar                = type('',(),{}) # 4DVar class
    varDA.fdvar.maxouter       = 1              # no. of outer loops for 4DVar
    varDA.fdvar.window         = 0.025          # length of the 4Dvar assimilation window
    varDA.fdvar.offset         = 0.5            # time offset: forecast from analysis to background time
    varDA.fdvar.nobstimes      = 2              # no. of evenly spaced obs. times in the window

# name and attributes of/in the output diagnostic file
diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = model.Name + '_hybDA_diag.nc4'
diag_file.attributes = {'model'       : model.Name,
                        'F'           : model.Par[0],
                        'dF'          : model.Par[1]-model.Par[0],
                        'dt'          : model.dt,
                        'ntimes'      : DA.ntimes,
                        'do_hybrid'   : int(DA.do_hybrid),
                        'hybrid_wght' : DA.hybrid_wght,
                        'hybrid_rcnt' : int(DA.hybrid_rcnt),
                        'Eupdate'     : ensDA.update,
                        'infl_meth'   : ensDA.inflation.infl_meth,
                        'infl_fac'    : ensDA.inflation.infl_fac,
                        'localize'    : int(ensDA.localization.localize),
                        'cov_cutoff'  : ensDA.localization.cov_cutoff,
                        'Vupdate'     : varDA.update,
                        'maxiter'     : varDA.minimization.maxiter,
                        'alpha'       : varDA.minimization.alpha,
                        'cg'          : int(varDA.minimization.cg),
                        'tol'         : varDA.minimization.tol}
if ( fdvar ):
    diag_file.attributes.update({'offset'    : varDA.fdvar.offset,
                                 'window'    : varDA.fdvar.window,
                                 'nobstimes' : int(varDA.fdvar.nobstimes),
                                 'maxouter'  : int(varDA.fdvar.maxouter)})

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = -1                # None == default | -N...-1 0 1...N
restart.filename = '../data/L96/ensDA_N=30/inf=1.06/L96_ensDA_diag-0.nc4'
