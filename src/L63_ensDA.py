#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L63_ensDA.py - cycle Ensemble DA on Lorenz 1963
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
###############################################################

###############################################################
global model
global Q, H, R
global DA, ensDA
global diag_file
global restart

model      = type('', (), {})              # model Class
model.Name = 'L63'                         # model name
model.Ndof = 3                             # model degrees of freedom
model.Par  = [10.0, 28.0, 8.0/3.0]         # model parameters
model.dt   = 1.0e-2                        # model time-step

Q = np.eye(model.Ndof)*0.0      # model error variance (covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(2.0)    # observation error covariance

DA        = type('', (), {})    # data assimilation Class
DA.nassim = 200                 # no. of assimilation cycles
DA.ntimes = 0.25                # do assimilation every ntimes non-dimensional time units
DA.t0     = 0.0                 # initial time

ensDA              = type('', (), {})  # ensemble data assimilation Class
ensDA.inflation    = type('', (), {})  # inflation Class
ensDA.localization = type('', (), {})  # localization Class
ensDA.update                  = 2      # DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 100    # number of ensemble members
ensDA.inflation.infl_meth     = 1      # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                       # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
ensDA.inflation.infl_fac      = 1.1    # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = True   # do localization
ensDA.localization.cov_cutoff = 1.0    # normalized covariance cutoff = cutoff / ( 2*normalized_dist)

# name and attributes of/in the output diagnostic file
diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = model.Name + '_ensDA_diag.nc4'
diag_file.attributes = {'model'       : str(model.Name),
                        'sigma'       : str(model.Par[0]),
                        'rho'         : str(model.Par[1]),
                        'beta'        : str(model.Par[2]),
                        'ntimes'      : str(DA.ntimes),
                        'dt'          : str(model.dt),
                        'Eupdate'     : str(ensDA.update),
                        'localize'    : str(int(ensDA.localization.localize)),
                        'cov_cutoff'  : str(ensDA.localization.cov_cutoff),
                        'infl_meth'   : str(ensDA.inflation.infl_meth),
                        'infl_fac'    : str(ensDA.inflation.infl_fac)}

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = None              # None == default | -1...-N 0 1...N
restart.filename = ''
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
    write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio = np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        exec('xs = integrate.odeint(%s, xt, DA.tanal, (model.Par,0.0))' % (model.Name))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            exec('xs = integrate.odeint(%s, xa, DA.tanal, (model.Par,0.0))' % (model.Name))
            Xb[:,m] = xs[-1,:].copy()

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA)

        # write diagnostics to disk
        write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio = evratio)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
