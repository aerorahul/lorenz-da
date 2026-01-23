#!/usr/bin/env python

###############################################################
# L96_param_truth.py - parameters for generating truth for L96
###############################################################

###############################################################
from module_IO import Container
from module_DA import DataAssim, EnsDataAssim
from module_Lorenz import Lorenz
import numpy
###############################################################

###############################################################
###############################################################

# insure the same sequence of random numbers EVERY TIME
numpy.random.seed(0)

model = Lorenz()
DA = DataAssim()

# Initialize Lorenz model class
Name = 'L96'              # model name
Ndof = 40                 # model degrees of freedom
Par = [8.0, 8.4]         # model parameters F, F+dF
dt = 1.0e-4             # model time-step
model.init(Name='L96', Ndof=40, Par=[8.0, 8.4], dt=1.e-4)

# Initialize Data Assimilation class
nassim = 2000                # no. of assimilation cycles
ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
maxouter = 1              # no. of outer loops
DA.init(nassim=nassim, ntimes=ntimes, maxouter=maxouter)
#DA.t0 = 0.0                 # initial time
# no. of obs to assimilate ( DA.Nobs <= model.Ndof)
#DA.Nobs = 20

# model error covariance ( covariance model is white for now )
Q = numpy.ones(model.Ndof)
Q = numpy.diag(Q) * 0.0

# obs operator ( eye(Ndof) gives identity obs )
H = numpy.ones(model.Ndof)
# H[::2] = numpy.nan
H = numpy.diag(H)

R = numpy.ones(model.Ndof)                   # observation error covariance
#R = R + numpy.random.rand(model.Ndof)
# R[1::2] = numpy.sqrt(2.0)
# R[1::4] = numpy.sqrt(3.0)
R = numpy.diag(R)

# name and attributes of/in the output diagnostic file
filename = model.Name + '_truthDA_diag.nc4'
attributes = {'model': model.Name,
              'F': model.Par[0],
              'dF': model.Par[1] - model.Par[0],
              'ntimes': DA.ntimes,
              'dt': model.dt}
diag_file = Container(filename=filename, attributes=attributes)

# restart conditions
time = None          # None == default | -N...-1 0 1...N
filename = 'L96_truth_diag.nc4'
restart = Container(time=time, filename=filename)

# ========== Clear unwanted parameters ==========
del numpy, Lorenz, Container
del Name, Ndof, Par, dt
del nassim, ntimes, maxouter
del filename, attributes
del time
