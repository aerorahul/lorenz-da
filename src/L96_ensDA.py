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
from   module_Lorenz import L96, plot_L96
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
global Ndof, F, dF, lab
global Q, H, R
global nassim, ntimes, dt, t0
global Eupdate, Nens, inflation, localization
global diag_fname, diag_fattr

Ndof = 40
F    = 8.0
dF   = 0.4
lab  = []
for j in range(0,Ndof): lab.append( 'x' + str(j+1) )

Q = np.eye(Ndof)*0.0            # model error variance (covariance model is white for now)
H = np.eye(Ndof)                # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(Ndof)*(1.0**2)       # observation error covariance

nassim = 2000                   # no. of assimilation cycles
ntimes = 0.05                   # do assimilation every ntimes non-dimensional time units
dt     = 1.0e-4                 # time-step
t0     = 0.0                    # initial time

Eupdate      = 2                # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens         = 30               # number of ensemble members
localize     = True             # do localization
cov_cutoff   = 1.0              # normalized covariance cutoff = cutoff / ( 2*normalized_dist)
localization = [localize, cov_cutoff]
infl_meth    = 1                # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac     = 1.06             # Depends on inflation method (see values in [] above)
inflation    = [infl_meth, infl_fac]

# name and attributes of/in the output diagnostic file
diag_fname = 'L96_ensDA_diag.nc4'
diag_fattr = {'F'           : str(F),
              'dF'          : str(dF),
              'ntimes'      : str(ntimes),
              'dt'          : str(dt),
              'Eupdate'     : str(Eupdate),
              'localize'    : str(int(localize)),
              'cov_cutoff'  : str(cov_cutoff),
              'infl_meth'   : str(infl_meth),
              'infl_fac'    : str(infl_fac)}
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble data assimilation options
    check_ensDA(Eupdate, inflation, localization)

    # initial setup from LE1998
    x0    = np.ones(Ndof) * F
    x0[0] = 1.001 * F

    # Make a copy of truth for plotting later
    xt = x0.copy()

    # populate initial ensemble analysis by perturbing true state
    [tmp, Xa] = np.meshgrid(np.ones(Nens),xt)
    pert = 0.001 * ( np.random.randn(Ndof,Nens) )
    Xa = Xa + pert

    # re-center the ensemble about initial true state
    xam = np.mean(Xa,axis=1)
    Xap = np.transpose(np.transpose(Xa) - xam)
    Xa  = np.transpose(xt + np.transpose(Xap))

    Xb = Xa.copy()

    print 'Cycling ON the attractor ...'

    ts = np.arange(t0,ntimes+dt,dt)     # time between assimilations

    # create diagnostic file
    create_diag(diag_fname, diag_fattr, Ndof, nens=Nens)
    write_diag(diag_fname, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio = np.NaN)

    for k in range(0, nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        xs = integrate.odeint(L96, xt, ts, (F,0.0))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,Nens):
            xa = Xa[:,m].copy()
            xs = integrate.odeint(L96, xa, ts, (F+dF,0.0))
            Xb[:,m] = xs[-1,:].copy()

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, Eupdate=Eupdate, inflation=inflation, localization=localization)

        # write diagnostics to disk
        write_diag(diag_fname, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio = evratio)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
