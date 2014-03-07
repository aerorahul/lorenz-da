#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_param_ensvarDA.py - parameters for Ensemble-Var DA on
#                         on L96
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import numpy
from module_Lorenz import Lorenz
from module_DA import DataAssim, VarDataAssim, EnsDataAssim
from module_IO import Container
###############################################################

# insure the same sequence of random numbers EVERY TIME
numpy.random.seed(0)

model = Lorenz()
DA    = DataAssim()
varDA = VarDataAssim()
ensDA = EnsDataAssim()

# Initialize Lorenz model class
Name = 'L96'              # model name
Ndof = 40                 # model degrees of freedom
Par  = [8.0, 8.4]         # model parameters
dt   = 1.0e-4             # model time-step
model.init(Name='L96',Ndof=40,Par=[8.0,8.4],dt=1.e-4)

# Initialize Data Assimilation class
nassim      = 1000           # no. of assimilation cycles
ntimes      = 2.0*0.05       # do assimilation every ntimes non-dimensional time units
maxouter    = 1              # no. of outer loops
do_hybrid   = True           # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
hybrid_rcnt = True           # True= re-center ensemble about varDA, False= free ensDA
hybrid_wght = 1.0            # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
DA.init(nassim=nassim,ntimes=ntimes,maxouter=maxouter,\
        do_hybrid=do_hybrid,hybrid_rcnt=hybrid_rcnt,hybrid_wght=hybrid_wght)

Q = numpy.ones(model.Ndof)                   # model error covariance ( covariance model is white for now )
Q = numpy.diag(Q) * 0.0

H = numpy.ones(model.Ndof)                   # obs operator ( eye(Ndof) gives identity obs )
#H[::2] = numpy.NaN
H = numpy.diag(H)

R = numpy.ones(model.Ndof)                   # observation error covariance
#R[1::2] = numpy.sqrt(2.0)
#R[1::4] = numpy.sqrt(3.0)
R = numpy.diag(R)

# Initialize Ensemble Data Assimilation class
update            = 2              # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens              = 20             # number of ensemble members
init_ens_infl_fac = 1.0            # inflate initial ensemble by init_ens_infl_fac
inflate           = 1              # inflation (0= None, 1= Multiplicative [1.01], 2= Additive [0.01],
                                               # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0])
infl_fac          = 1.06           # Depends on inflation method (see values in [] above)
localize          = 1              # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
cov_cutoff        = 0.0625         # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
cov_trunc         = model.Ndof     # truncate localization matrix (cov_trunc <= model.Ndof)
ensDA.init(model,DA,\
           update=update,Nens=Nens,init_ens_infl_fac=init_ens_infl_fac,\
           inflate=inflate,infl_fac=infl_fac,\
           localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc)

# Initialize Variational Data Assimilation class
update       = 2              # variational-based DA method (1= 3Dvar; 2= 4Dvar)
precondition = 1              # precondition before minimization [0= None; 1= sqrtB; 2= FullB]
maxiter      = 100            # maximum iterations for minimization
tol          = 1e-4           # tolerance to end the variational minimization iteration
inflate      = True           # inflate [ > 1.0 ] / deflate [ < 1.0 ] static covariance
infl_fac     = 1.85           # inflate static covariance
infl_adp     = True           # inflate adaptively (cuts inflation as a function  of OL)
localize     = 1              # localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
cov_cutoff   = 0.0625         # normalized covariance cutoff = cutoff / ( 2*normalized_dist )
cov_trunc    = model.Ndof     # truncate localization matrix (cov_trunc <= model.Ndof)
window       = 0.75           # length of the assimilation window
offset       = 0.25           # time offset: forecast from analysis to background time
nobstimes    = 4              # no. of evenly spaced obs. times in the window
varDA.init(model,DA,\
           update=update,precondition=precondition,\
           maxiter=maxiter,tol=tol,\
           inflate=inflate,infl_fac=infl_fac,infl_adp=infl_adp,\
           localize=localize,cov_cutoff=cov_cutoff,cov_trunc=cov_trunc,\
           window=window,offset=offset,nobstimes=nobstimes)

# Initialize diagnostic file class
filename   = model.Name + '_ensvarDA_diag.nc4'
attributes = {'model'       : model.Name,
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
              'precondition': varDA.precondition,
              'maxiter'     : varDA.minimization.maxiter,
              'tol'         : varDA.minimization.tol,
              'Vinflate'    : int(varDA.inflation.inflate),
              'Vinfl_fac'   : varDA.inflation.infl_fac,
              'Vinfl_adp'   : int(varDA.inflation.infl_adp),
              'Vlocalize'   : varDA.localization.localize,
              'Vcov_cutoff' : varDA.localization.cov_cutoff,
              'Vcov_trunc'  : varDA.localization.cov_trunc,
              'offset'      : varDA.fdvar.offset,
              'window'      : varDA.fdvar.window,
              'nobstimes'   : varDA.fdvar.nobstimes}
diag_file = Container(filename=filename,attributes=attributes)

# restart conditions
time     = -1              # None == default | -N...-1 0 1...N
filename = 'L96_ensDA_diag.nc4'
restart = Container(time=time,filename=filename)

# ========== Clear unwanted parameters ==========
del numpy, Lorenz, DataAssim, EnsDataAssim, VarDataAssim, Container
del Name, Ndof, Par, dt
del nassim, ntimes, maxouter
del do_hybrid, hybrid_rcnt, hybrid_wght
del update, \
    inflate, infl_fac, infl_adp, \
    localize, cov_cutoff, cov_trunc
del Nens, init_ens_infl_fac
del precondition, \
    maxiter, tol, \
    window, offset, nobstimes
del filename, attributes
del time
