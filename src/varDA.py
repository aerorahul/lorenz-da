#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# varDA.py - Variational DA on Lorenz 63 or Lorenz & Emanuel 96
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
from   module_Lorenz import *
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import *
from   param_varDA   import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid variational data assimilation options
    check_DA(DA)
    check_varDA(varDA)

    # get IC's
    [xt, xa] = get_IC(model, restart)
    xb = xa.copy()

    # load fixed background error covariance matrix; generated by 'model.Name'_stats.py
    nc = Dataset(model.Name + '_climo_B.nc4','r')
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

    else:
        # time between assimilations
        DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_diag(diag_file, model.Ndof)
    write_diag(diag_file.filename, 0, xt, xb, xa, np.dot(H,xt), np.diag(H), np.diag(R), niters=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        if ( fdvar ):
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xt, varDA.fdvar.tfore, (model.Par,   0.0))' % (model.Name))
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xt, varDA.fdvar.tfore, (model.Par[0],0.0))' % (model.Name))
            xt = xs[varDA.fdvar.ta,:].copy()
        else:
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xt, DA.tanal, (model.Par,   0.0))' % (model.Name,))
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xt, DA.tanal, (model.Par[0],0.0))' % (model.Name,))
            xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
        ver = xt.copy()
        if ( fdvar ):
            ywin = np.zeros((varDA.fdvar.nobstimes,model.Ndof))
            for i in range(0,varDA.fdvar.nobstimes):
                ywin[i,:] = np.dot(H,xs[varDA.fdvar.twind_obsIndex[i]+varDA.fdvar.tb,:] + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
            oind = np.where(varDA.fdvar.twind_obsIndex + varDA.fdvar.tb == varDA.fdvar.ta)
            if ( len(oind[0]) != 0 ): y = ywin[oind[0][0],:].copy()

        # advance analysis with the full nonlinear model
        if ( fdvar ):
            # step to the beginning of the assimilation window (varDA.fdvar.tbkgd)
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xa, varDA.fdvar.tbkgd, (model.Par,   0.0))' % (model.Name))
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xa, varDA.fdvar.tbkgd, (model.Par[1],0.0))' % (model.Name))
        else:
            # step to the next assimilation time (DA.tanal)
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xa, DA.tanal, (model.Par,   0.0))' % (model.Name))
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xa, DA.tanal, (model.Par[1],0.0))' % (model.Name))
        xb = xs[-1,:].copy()

        # update step
        if ( fdvar ): xa, Ac, niters = update_varDA(xb, Bc, ywin, R, H, varDA, model=model)
        else:         xa, Ac, niters = update_varDA(xb, Bc, y,    R, H, varDA)

        # if doing 4Dvar, step to the next assimilation time from the beginning of assimilation window
        if ( fdvar ):
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xb, varDA.fdvar.tanal, (model.Par,   0.0))' % (model.Name))
                xb = xs[-1,:].copy()
                exec('xs = integrate.odeint(%s, xa, varDA.fdvar.tanal, (model.Par,   0.0))' % (model.Name))
                xa = xs[-1,:].copy()
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xb, varDA.fdvar.tanal, (model.Par[1],0.0))' % (model.Name))
                xb = xs[-1,:].copy()
                exec('xs = integrate.odeint(%s, xa, varDA.fdvar.tanal, (model.Par[1],0.0))' % (model.Name))
                xa = xs[-1,:].copy()

        # write diagnostics to disk
        write_diag(diag_file.filename, k+1, ver, xb, xa, y, np.diag(H), np.diag(R), niters=niters)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
