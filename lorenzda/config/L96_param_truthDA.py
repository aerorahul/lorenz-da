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
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import numpy  as np
###############################################################

# insure the same sequence of random numbers EVERY TIME
np.random.seed(0)

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 8.4]         # model parameters F, F+dF
model.dt   = 1.0e-4             # model time-step

DA        = type('', (), {})    # data assimilation Class
DA.nassim = 2000                # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time
DA.Nobs   = 20                  # no. of obs to assimilate ( DA.Nobs <= model.Ndof)

H = np.ones(model.Ndof)                   # obs operator ( eye(Ndof) gives identity obs )
if ( DA.Nobs != model.Ndof ):
    index = np.arange(model.Ndof)
    np.random.shuffle(index)
    H[index[:-DA.Nobs]] = np.NaN
H = np.diag(H)

R = np.ones(model.Ndof)                   # observation error covariance
R = R + np.random.rand(model.Ndof)
R = np.diag(R)

# name and attributes of/in the output diagnostic file
truth_file            = type('', (), {})  # diagnostic file Class
truth_file.filename   = model.Name + '_truthDA_diag.nc4'
truth_file.attributes = {'model'       : model.Name,
                         'F'           : model.Par[0],
                         'dF'          : model.Par[1]-model.Par[0],
                         'ntimes'      : DA.ntimes,
                         'dt'          : model.dt}

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
