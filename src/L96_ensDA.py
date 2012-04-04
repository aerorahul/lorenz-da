#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_ensDA.py - cycle Ensemble DA on Lorenz & Emanuel 1998
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
global Q, H, R
global DA
global ensDA
global diag_fname, diag_fattr
global restart_state, restart_file

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 0.4]         # model parameters F, dF
model.dt   = 1.0e-4             # model time-step

Q = np.eye(model.Ndof)*0.0      # model error variance (covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

DA        = type('', (), {})    # data assimilation Class
DA.nassim = 2000                # no. of assimilation cycles
DA.ntimes = 0.05                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

ensDA        = type('', (), {}) # ensemble data assimilation Class
ensDA.update = 2                # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens   = 30               # number of ensemble members
infl_meth    = 1                # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac     = 1.06             # Depends on inflation method (see values in [] above)
localize     = True             # do localization
cov_cutoff   = 1.0              # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
ensDA.inflation    = [infl_meth, infl_fac]
ensDA.localization = [localize, cov_cutoff]

# name and attributes of/in the output diagnostic file
diag_fname = 'L96_ensDA_diag.nc4'
diag_fattr = {'F'           : str(model.Par[0]),
              'dF'          : str(model.Par[1]),
              'ntimes'      : str(DA.ntimes),
              'dt'          : str(model.dt),
              'Eupdate'     : str(ensDA.update),
              'localize'    : str(int(ensDA.localization[0])),
              'cov_cutoff'  : str(ensDA.localization[1]),
              'infl_meth'   : str(ensDA.inflation[0]),
              'infl_fac'    : str(ensDA.inflation[1])}

# restart conditions ( state [< -1 | == -1 | > -1], filename)
restart_state = -2
restart_file  = ''
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble data assimilation options
    check_ensDA(ensDA)

    # get IC's
    [xt, Xa] = get_IC(model=model, restart_state=restart_state, restart_file=restart_file, Nens=ensDA.Nens)
    Xb = Xa.copy()

    print 'Cycling ON the attractor ...'

    ts = np.arange(DA.t0,DA.ntimes+model.dt,model.dt)     # time between assimilations

    # create diagnostic file
    create_diag(diag_fname, diag_fattr, model.Ndof, nens=ensDA.Nens)
    write_diag(diag_fname, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio = np.NaN)

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        xs = integrate.odeint(L96, xt, ts, (model.Par[0],0.0))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            xs = integrate.odeint(L96, xa, ts, (model.Par[0]+model.Par[1],0.0))
            Xb[:,m] = xs[-1,:].copy()

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA)

        # write diagnostics to disk
        write_diag(diag_fname, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio = evratio)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
