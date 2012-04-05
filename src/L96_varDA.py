#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_varDA.py - cycle variational DA on Lorenz & Emanuel 1998
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
from   matplotlib    import pyplot
from   netCDF4       import Dataset
from   module_Lorenz import L96, plot_L96, get_IC
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
global model
global A, Q, H, R
global DA
global varDA
global diag_file
global restart

model      = type('',(),{})  # model Class
model.Name = 'L96'           # model name
model.Ndof = 40              # model degrees of freedom
model.Par  = [8.0, 0.4]      # model parameters F, dF
model.dt   = 1.0e-4          # model time-step

A = np.eye(model.Ndof)          # initial analysis error covariance
Q = np.eye(model.Ndof)*0.0      # model error covariance ( covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

DA        = type('',(),{})      # DA class
DA.nassim = 2000                # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

varDA                      = type('',(),{}) # VarDA class
varDA.update               = 1              # DA method (1= 3Dvar; 2= 4Dvar)
varDA.minimization         = type('',(),{}) # minimization class
varDA.minimization.maxiter = 1000           # maximum iterations
varDA.minimization.alpha   = 4e-4           # size of step in direction of normalized J
varDA.minimization.cg      = True           # True = Use conjugate gradient; False = Perform line search
varDA.minimization.tol     = 1e-5           # tolerance to end the variational minimization iteration

if ( (varDA.update == 2) or (varDA.update == 4) ): fdvar = True
else:                                              fdvar = False

if ( fdvar ):
    varDA.fdvar                = type('',(),{}) # 4DVar class
    varDA.fdvar.maxouter       = 1              # no. of outer loops for 4DVar
    varDA.fdvar.window         = DA.ntimes      # length of the 4Dvar assimilation window
    varDA.fdvar.offset         = 0.5            # time offset: forecast from analysis to background time
    varDA.fdvar.nobstimes      = 11             # no. of evenly spaced obs. times in the window

diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = 'L96_varDA_diag.nc4'
diag_file.attributes = {'F'       : str(model.Par[0]),
                        'dF'      : str(model.Par[1]),
                        'ntimes'  : str(DA.ntimes),
                        'dt'      : str(model.dt),
                        'Vupdate' : str(varDA.update),
                        'maxiter' : str(varDA.minimization.maxiter),
                        'alpha'   : str(varDA.minimization.alpha),
                        'cg'      : str(int(varDA.minimization.cg)),
                        'tol'     : str(int(varDA.minimization.tol))}
if ( fdvar ):
    diag_file.attributes.update({'offset'    : str(varDA.fdvar.offset),
                                 'window'    : str(varDA.fdvar.window),
                                 'nobstimes' : str(int(varDA.fdvar.nobstimes)),
                                 'maxouter'  : str(int(varDA.fdvar.maxouter))})

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = 0                 # 0 | None == default, 1...N | -1...-N
restart.filename = ''

###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid variational data assimilation options
    check_varDA(varDA)

    # get IC's
    [xt, xa] = get_IC(model, restart)
    xb = xa.copy()

    # load fixed background error covariance matrix; generated by L96_stats.py and compute inverse
    nc = Dataset('L96_climo_B.nc4','r')
    Bc = nc.variables['B'][:]
    nc.close()

    if ( fdvar ):
        # check length of assimilation window
        if ( varDA.fdvar.offset * DA.ntimes + varDA.fdvar.window - DA.ntimes < 0.0 ):
            print 'assimilation window is too short'
            sys.exit(2)

        # time index from analysis to ... background, next analysis, end of window, window
        varDA.fdvar.tb = np.int(np.rint(varDA.fdvar.offset * DA.ntimes/model.dt))
        varDA.fdvar.ta = np.int(np.rint(DA.ntimes/model.dt))
        varDA.fdvar.tf = np.int(np.rint((varDA.fdvar.offset * DA.ntimes + varDA.fdvar.window)/model.dt))
        varDA.fdvar.tw = varDA.fdvar.tf - varDA.fdvar.tb

        # time vector from analysis to ... background, next analysis, end of window, window
        varDA.fdvar.tbkgd = np.linspace(DA.t0,varDA.fdvar.tb,   varDA.fdvar.tb   +1) * model.dt
        varDA.fdvar.tanal = np.linspace(DA.t0,varDA.fdvar.ta-varDA.fdvar.tb,varDA.fdvar.ta-varDA.fdvar.tb+1) * model.dt
        varDA.fdvar.tfore = np.linspace(DA.t0,varDA.fdvar.tf,   varDA.fdvar.tf   +1) * model.dt
        varDA.fdvar.twind = np.linspace(DA.t0,varDA.fdvar.tw,   varDA.fdvar.tw   +1) * model.dt

        # time vector, interval, indices of observations
        varDA.fdvar.twind_obsInterval = varDA.fdvar.tw / (varDA.fdvar.nobstimes-1)
        varDA.fdvar.twind_obsTimes    = varDA.fdvar.twind[::varDA.fdvar.twind_obsInterval]
        varDA.fdvar.twind_obsIndex    = np.array(np.rint(varDA.fdvar.twind_obsTimes / model.dt), dtype=int)

    print 'Cycling ON the attractor ...'

    if ( not fdvar ):
        DA.tanal = np.arange(DA.t0,DA.ntimes+model.dt,model.dt)  # time between assimilations

    # create diagnostic file
    create_diag(diag_file, model.Ndof)
    write_diag(diag_file.filename, 0, xt, xb, xa, np.dot(H,xt), H, np.diag(R), niters=np.NaN)

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        if ( fdvar ):
            xs = integrate.odeint(L96, xt, varDA.fdvar.tfore, (model.Par[0],0.0))
            xt = xs[varDA.fdvar.ta,:].copy()
        else:
            xs = integrate.odeint(L96, xt, DA.tanal, (model.Par[0],0.0))
            xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        if ( fdvar ):
            y = np.zeros((varDA.fdvar.nobstimes,model.Ndof))
            for i in range(0,varDA.fdvar.nobstimes):
                y[i,:] = np.dot(H,xs[varDA.fdvar.twind_obsIndex[i]+varDA.fdvar.tb,:]) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
            ver = xt.copy()
            ya  = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        else:
            y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
            ver = xt.copy()

        if ( fdvar ):
            # step to the beginning of the assimilation window
            xs = integrate.odeint(L96, xa, varDA.fdvar.tbkgd, (model.Par[0]+model.Par[1],0.0))
            xb = xs[-1,:].copy()
        else:
            # step to the next assimilation time
            xs = integrate.odeint(L96, xa, DA.tanal, (model.Par[0]+model.Par[1],0.0))
            xb = xs[-1,:].copy()

        # update step
        xa, A, niters = update_varDA(xb, Bc, y, R, H, varDA, model=model)

        # write diagnostics to disk
        if ( fdvar ):
            write_diag(diag_file.filename, k+1, ver, xb, xa, ya, H, np.diag(R), niters=niters)
        else:
            write_diag(diag_file.filename, k+1, ver, xb, xa, y,  H, np.diag(R), niters=niters)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
