#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# ensDA.py - Ensemble DA on Lorenz 63 or Lorenz & Emanuel 96
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
from   param_ensDA   import *
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble data assimilation options
    check_DA(DA)
    check_ensDA(ensDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xb = Xa.copy()

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens)
    write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), np.diag(H), np.diag(R), evratio = np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        if   ( model.Name == 'L63' ):
            exec('xs = integrate.odeint(%s, xt, DA.tanal, (model.Par,   0.0))' % (model.Name))
        elif ( model.Name == 'L96' ):
            exec('xs = integrate.odeint(%s, xt, DA.tanal, (model.Par[0],0.0))' % (model.Name))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt + np.random.randn(model.Ndof) * np.sqrt(np.diag(R)))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            if   ( model.Name == 'L63' ):
                exec('xs = integrate.odeint(%s, xa, DA.tanal, (model.Par,   0.0))' % (model.Name))
            elif ( model.Name == 'L96' ):
                exec('xs = integrate.odeint(%s, xa, DA.tanal, (model.Par[1],0.0))' % (model.Name))
            Xb[:,m] = xs[-1,:].copy()

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA)

        # write diagnostics to disk
        write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, np.diag(H), np.diag(R), evratio = evratio)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
