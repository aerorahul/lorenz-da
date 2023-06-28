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
from module_IO import Container
from module_DA import DataAssim, EnsDataAssim
from module_Lorenz import Lorenz
import numpy
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################
###############################################################

# insure the same sequence of random numbers EVERY TIME
numpy.random.seed(0)

model = Lorenz()
DA = DataAssim()
ensDA = EnsDataAssim()

# Initialize Lorenz model class
Name = 'L96'              # model name
Ndof = 40                 # model degrees of freedom
Par = [8.0, 8.4]         # model parameters
dt = 1.0e-4             # model time-step
model.init(Name='L96', Ndof=40, Par=[8.0, 8.4], dt=1.e-4)


# Initialize Data Assimilation class
nassim = 1000           # no. of assimilation cycles
ntimes = 1.0 * 0.05       # do assimilation every ntimes non-dimensional time units
maxouter = 1              # no. of outer loops
DA.init(nassim=nassim, ntimes=ntimes, maxouter=maxouter)

# model error covariance ( covariance model is white for now )
Q = numpy.ones(model.Ndof)
Q = numpy.diag(Q) * 0.0

# obs operator ( eye(Ndof) gives identity obs )
H = numpy.ones(model.Ndof)
# H[::2] = numpy.NaN
H = numpy.diag(H)

R = numpy.ones(model.Ndof)                   # observation error covariance
# R[1::2] = numpy.sqrt(2.0)
# R[1::4] = numpy.sqrt(3.0)
R = numpy.diag(R)

# ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
update = 2
Nens = 20             # number of ensemble members
init_ens_infl_fac = 1.0            # inflate initial ensemble by init_ens_infl_fac
# inflation (0= None, 1= Multiplicative [1.01], 2= Additive [0.01],
inflate = 1
# 3= Cov. Relax [0.25], 4= Spread Restoration [1.0])
# Depends on inflation method (see values in [] above)
infl_fac = 1.06
# localization (0= None, 1= Gaspari-Cohn, 2= Boxcar, 3= Ramped)
localize = 1
# normalized covariance cutoff = cutoff / ( 2*normalized_dist)
cov_cutoff = 0.0625
# truncate localization matrix (cov_trunc <= model.Ndof)
cov_trunc = model.Ndof
ensDA.init(model, DA,
           update=update, Nens=Nens, init_ens_infl_fac=init_ens_infl_fac,
           inflate=inflate, infl_fac=infl_fac,
           localize=localize, cov_cutoff=cov_cutoff, cov_trunc=cov_trunc)

# name and attributes of/in the output diagnostic file
filename = model.Name + '_ensDA_diag_out.nc4'
attributes = {'model': model.Name,
              'F': model.Par[0],
              'dF': model.Par[1] - model.Par[0],
              'dt': model.dt,
              'ntimes': DA.ntimes,
              'Eupdate': ensDA.update,
              'Elocalize': ensDA.localization.localize,
              'Ecov_cutoff': ensDA.localization.cov_cutoff,
              'Ecov_trunc': ensDA.localization.cov_trunc,
              'Einflate': ensDA.inflation.inflate,
              'Einfl_fac': ensDA.inflation.infl_fac}
diag_file = Container(filename=filename, attributes=attributes)

# restart conditions
time = -1            # None == default | -N...-1 0 1...N
filename = 'L96_ensDA_diag.nc4'
restart = Container(time=time, filename=filename)

# ========== Clear unwanted parameters ==========
del numpy, Lorenz, DataAssim, EnsDataAssim, Container
del Name, Ndof, Par, dt
del nassim, ntimes, maxouter
del update, Nens, init_ens_infl_fac, \
    inflate, infl_fac, \
    localize, cov_cutoff, cov_trunc
del filename, attributes
del time
