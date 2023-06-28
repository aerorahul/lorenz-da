#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_param_truth.py - parameters for generating truth for L96
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

model = type('', (), {})      # model Class
model.Name = 'L63'                 # model name
model.Ndof = 3                     # model degrees of freedom
model.Par = [10.0, 28.0, 8.0 / 3.0]  # model parameters
model.dt = 1.0e-2                # model time-step

DA = type('', (), {})    # data assimilation Class
DA.nassim = 200                 # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0 = 0.0                 # initial time

# model error covariance ( covariance model is white for now )
Q = np.ones(model.Ndof)
Q = np.diag(Q) * 0.0

# obs operator ( eye(Ndof) gives identity obs )
H = np.ones(model.Ndof)
H = np.diag(H)

R = np.ones(model.Ndof)                # observation error covariance
R = 2.0 * R
R = np.diag(R)

# name and attributes of/in the output diagnostic file
truth_file = type('', (), {})  # diagnostic file Class
truth_file.filename = model.Name + '_truthDA_diag.nc4'
truth_file.attributes = {'model': model.Name,
                         'sigma': model.Par[0],
                         'rho': model.Par[1],
                         'beta': model.Par[2],
                         'ntimes': DA.ntimes,
                         'dt': model.dt}

# restart conditions
restart = type('', (), {})  # restart initial conditions Class
restart.time = None              # None == default | -1...-N 0 1...N
restart.filename = ''
