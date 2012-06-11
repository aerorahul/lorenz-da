#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_truthDA.py - generate truth and observations about truth
#                  for DA on Lorenz & Emanuel 1998
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2012, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import sys
import numpy         as     np
from   scipy         import integrate, io
from   netCDF4       import Dataset
from   module_Lorenz import *
from   module_IO     import *
###############################################################

###############################################################
global model
global H, R
global DA
global truth_file
global restart

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 0.4]         # model parameters F, dF
model.dt   = 1.0e-4             # model time-step

H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

DA        = type('', (), {})    # data assimilation Class
DA.nassim = 2000                # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

# name and attributes of/in the output diagnostic file
truth_file            = type('', (), {})  # diagnostic file Class
truth_file.filename   = model.Name + '_truthDA_diag.nc4'
truth_file.attributes = {'F'           : str(model.Par[0]),
                         'dF'          : str(model.Par[1]),
                         'ntimes'      : str(DA.ntimes),
                         'dt'          : str(model.dt)}

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # get IC's
    [xt,_] = get_IC(model, restart)

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_truth(truth_file, model.Ndof)
    write_truth(truth_file.filename, 0, xt, np.dot(H,xt), H, np.diag(R))

    print 'running ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        exec('xs = integrate.odeint(%s, xt, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))

        # write diagnostics to disk
        write_truth(truth_file.filename, k+1, xt, y, H, np.diag(R))

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
