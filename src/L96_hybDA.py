#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# L96_hybDA.py - cycle Hybrid DA on Lorenz & Emanuel 1998
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
global DA, ensDA, varDA
global diag_file
global restart

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 0.4]         # model parameters F, dF
model.dt   = 1.0e-4             # model time-step

Q = np.eye(model.Ndof)*0.0      # model error variance (covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

DA             = type('', (), {}) # data assimilation Class
DA.nassim      = 2000             # no. of assimilation cycles
DA.ntimes      = 0.05             # do assimilation every ntimes non-dimensional time units
DA.t0          = 0.0              # initial time
DA.do_hybrid   = True             # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
DA.hybrid_wght = 0.0              # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
DA.hybrid_rcnt = False            # True= re-center ensemble about varDA, False= free ensDA

ensDA              = type('', (), {})  # ensemble data assimilation Class
ensDA.inflation    = type('', (), {})  # inflation Class
ensDA.localization = type('', (), {})  # localization Class
ensDA.update                  = 2      # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
ensDA.Nens                    = 30     # number of ensemble members
ensDA.inflation.infl_meth     = 1      # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                       # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
ensDA.inflation.infl_fac      = 1.06   # Depends on inflation method (see values in [] above)
ensDA.localization.localize   = True   # do localization
ensDA.localization.cov_cutoff = 1.0    # normalized covariance cutoff = cutoff / ( 2*normalized_dist)

varDA                      = type('', (), {}) # variational data assimilation Class
varDA.minimization         = type('', (), {}) # minimization Class
varDA.update               = 1                # variational-based DA method (1 = 3Dvar; 2= 4Dvar)
varDA.minimization.maxiter = 1000             # maximum iterations for minimization
varDA.minimization.alpha   = 4e-4             # size of step in direction of normalized J
varDA.minimization.cg      = True             # True = Use conjugate gradient; False = Perform line search
varDA.minimization.tol     = 1e-4             # True = Use conjugate gradient; False = Perform line search

# name and attributes of/in the output diagnostic file
diag_file            = type('', (), {})  # diagnostic file Class
diag_file.filename   = model.Name + '_hybDA_diag.nc4'
diag_file.attributes = {'F'           : str(model.Par[0]),
                        'dF'          : str(model.Par[1]),
                        'dt'          : str(model.dt),
                        'ntimes'      : str(DA.ntimes),
                        'do_hybrid'   : str(int(DA.do_hybrid)),
                        'hybrid_wght' : str(DA.hybrid_wght),
                        'hybrid_rcnt' : str(int(DA.hybrid_rcnt)),
                        'Eupdate'     : str(ensDA.update),
                        'infl_meth'   : str(ensDA.inflation.infl_meth),
                        'infl_fac'    : str(ensDA.inflation.infl_fac),
                        'localize'    : str(int(ensDA.localization.localize)),
                        'cov_cutoff'  : str(ensDA.localization.cov_cutoff),
                        'Vupdate'     : str(varDA.update),
                        'maxiter'     : str(varDA.minimization.maxiter),
                        'alpha'       : str(varDA.minimization.alpha),
                        'cg'          : str(int(varDA.minimization.cg)),
                        'tol'         : str(varDA.minimization.tol)}

# restart conditions
restart          = type('', (), {})  # restart initial conditions Class
restart.time     = 0                 # 0 | None == default, 1...N | -1...-N
restart.filename = ''
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble and variational data assimilation options
    check_ensDA(ensDA)
    check_varDA(varDA)

    # get IC's
    [xt, Xa] = get_IC(model, restart, Nens=ensDA.Nens)
    Xb = Xa.copy()
    if ( DA.do_hybrid ):
        xac = np.mean(Xa,axis=1)
        xbc = np.mean(Xb,axis=1)

    if ( DA.do_hybrid ):
        print 'load climatological covariance ...'
        nc = Dataset(model.Name + '_climo_B.nc4','r')
        Bs = nc.variables['B'][:]
        nc.close()

    # time between assimilations
    DA.tanal = model.dt * np.linspace(DA.t0,np.rint(DA.ntimes/model.dt),np.int(np.rint(DA.ntimes/model.dt)+1))

    # create diagnostic file
    create_diag(diag_file, model.Ndof, nens=ensDA.Nens, hybrid=DA.do_hybrid)
    if ( DA.do_hybrid ):
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), central_prior=xbc, central_posterior=xac, evratio=np.NaN, niters=np.NaN)
    else:
        write_diag(diag_file.filename, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio=np.NaN)

    print 'Cycling ON the attractor ...'

    for k in range(0, DA.nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        exec('xs = integrate.odeint(%s, xt, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,ensDA.Nens):
            xa = Xa[:,m].copy()
            exec('xs = integrate.odeint(%s, xa, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
            Xb[:,m] = xs[-1,:].copy()

        # compute background error covariance from the ensemble
        if ( DA.do_hybrid ): Be = np.cov(Xb, ddof=1)

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, ensDA)

        if ( DA.do_hybrid ):
            # advance central analysis with the full nonlinear model
            exec('xs = integrate.odeint(%s, xac, DA.tanal, (%f,0.0))' % (model.Name, model.Par[0]+model.Par[1]))
            xbc = xs[-1,:].copy()

            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Bc = (1.0 - DA.hybrid_wght) * Bs + DA.hybrid_wght * Be

            # update the central background
            xac, Ac, niters = update_varDA(xbc, Bc, y, R, H, varDA)

        # write diagnostics to disk before recentering
        if ( DA.do_hybrid ):
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), central_prior=xbc, central_posterior=xac, evratio=evratio, niters=niters)
        else:
            write_diag(diag_file.filename, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio=evratio)

        # recenter ensemble about central analysis
        if ( DA.do_hybrid ):
            if ( DA.hybrid_rcnt ): Xa = np.transpose(np.transpose(Xa) - np.mean(Xa,axis=1) + xac)

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
