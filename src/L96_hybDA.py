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
from   module_Lorenz import L96, plot_L96, get_IC
from   module_DA     import *
from   module_IO     import *
from   plot_stats    import *
###############################################################

###############################################################
global model
global Q, H, R
global nassim, ntimes, dt, t0
global Eupdate, Nens, inflation, localization
global Vupdate, minimization
global do_hybrid, hybrid_wght, hybrid_rcnt
global diag_fname, diag_fattr
global restart_state, restart_file

model      = type('', (), {})   # model Class
model.Name = 'L96'              # model name
model.Ndof = 40                 # model degrees of freedom
model.Par  = [8.0, 0.4]         # model parameters F, dF

Q = np.eye(model.Ndof)*0.0      # model error variance (covariance model is white for now)
H = np.eye(model.Ndof)          # obs operator ( eye(Ndof) gives identity obs )
R = np.eye(model.Ndof)*(1.0**2) # observation error covariance

nassim = 2000                   # no. of assimilation cycles
ntimes = 0.05                   # do assimilation every ntimes non-dimensional time units
dt     = 1.0e-4                 # time-step
t0     = 0.0                    # initial time

Eupdate      = 2                # ensemble-based DA method (0= No Assim, 1= EnKF; 2= EnSRF; 3= EAKF)
Nens         = 30               # number of ensemble members
localize     = True             # do localization
cov_cutoff   = 1.0              # cov_cutoff = cutoff_dist / ( 2 * (normalization_dist==Ndof) )
localization = [localize, cov_cutoff]
infl_meth    = 1                # inflation (1= Multiplicative [1.01], 2= Additive [0.01],
                                # 3= Cov. Relax [0.25], 4= Spread Restoration [1.0], 5= Adaptive)
infl_fac     = 1.06             # Depends on inflation method (see values in [] above)
inflation    = [infl_meth, infl_fac]

Vupdate = 1                     # variational-based DA method (1= 3Dvar; 2= 4Dvar)
maxiter = 1000                  # maximum iterations for minimization
alpha   = 4e-4                  # size of step in direction of normalized J
cg      = True                  # True = Use conjugate gradient; False = Perform line search
minimization = [maxiter, alpha, cg]

do_hybrid   = True              # True= run hybrid (varDA + ensDA) mode, False= run ensDA mode
hybrid_wght = 0.0               # weight for hybrid (0.0= Bstatic; 1.0= Bensemble)
hybrid_rcnt = False              # True= re-center ensemble about varDA, False= free ensDA

# name and attributes of/in the output diagnostic file
diag_fname = 'L96_hybDA_diag.nc4'
diag_fattr = {'F'           : str(model.Par[0]),
              'dF'          : str(model.Par[1]),
              'ntimes'      : str(ntimes),
              'dt'          : str(dt),
              'Eupdate'     : str(Eupdate),
              'localize'    : str(int(localize)),
              'cov_cutoff'  : str(cov_cutoff),
              'infl_meth'   : str(infl_meth),
              'infl_fac'    : str(infl_fac),
              'Vupdate'     : str(Vupdate),
              'maxiter'     : str(maxiter),
              'alpha'       : str(alpha),
              'cg'          : str(int(cg)),
              'do_hybrid'   : str(int(do_hybrid)),
              'hybrid_wght' : str(hybrid_wght),
              'hybrid_rcnt' : str(int(hybrid_rcnt))}

# restart conditions ( state [< -1 | == -1 | > -1], filename)
restart_state = -1
restart_file  = 'L96_hybDA_diag.nc4'
###############################################################

###############################################################
def main():

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    # check for valid ensemble and variational data assimilation options
    check_ensDA(Eupdate, inflation, localization)
    check_varDA(Vupdate)

    # get IC's
    [xt, Xa] = get_IC(model=model, restart_state=restart_state, restart_file=restart_file, Nens=Nens)
    Xb  = Xa.copy()
    xam = xt.copy()
    xbm = xam.copy()

    if ( do_hybrid ):
        print 'load climatological covariance ...'
        nc = Dataset('L96_climo_B.nc4','r')
        Bs = nc.variables['B'][:]
        nc.close()

    print 'Cycling ON the attractor ...'

    ts = np.arange(t0,ntimes+dt,dt)     # time between assimilations

    # create diagnostic file
    create_diag(diag_fname, diag_fattr, model.Ndof, nens=Nens, hybrid=do_hybrid)
    if ( do_hybrid ):
        write_diag(diag_fname, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), central_prior=xbm, central_posterior=xam, evratio=np.NaN, niters=np.NaN)
    else:
        write_diag(diag_fname, 0, xt, np.transpose(Xb), np.transpose(Xa), np.dot(H,xt), H, np.diag(R), evratio=np.NaN)

    for k in range(0, nassim):

        print '========== assimilation time = %5d ========== ' % (k+1)

        # advance truth with the full nonlinear model
        xs = integrate.odeint(L96, xt, ts, (model.Par[0],0.0))
        xt = xs[-1,:].copy()

        # new observations from noise about truth; set verification values
        y   = np.dot(H,xt) + np.random.randn(model.Ndof) * np.sqrt(np.diag(R))
        ver = xt.copy()

        # advance analysis ensemble with the full nonlinear model
        for m in range(0,Nens):
            xa = Xa[:,m].copy()
            xs = integrate.odeint(L96, xa, ts, (model.Par[0]+model.Par[1],0.0))
            Xb[:,m] = xs[-1,:].copy()

        # advance central analysis with the full nonlinear model
        if ( do_hybrid ):
            xs = integrate.odeint(L96, xam, ts, (model.Par[0]+model.Par[1],0.0))
            xbc = xs[-1,:].copy()

        # compute background ensemble mean and perturbations from the mean
        xbm = np.mean(Xb,axis=1)
        Xbp = np.transpose(np.transpose(Xb) - xbm)

        # compute background error covariance from the ensemble
        B = np.dot(Xbp,np.transpose(Xbp)) / (Nens - 1)

        # update ensemble (mean and perturbations)
        Xa, evratio = update_ensDA(Xb, y, R, H, Eupdate=Eupdate, inflation=inflation, localization=localization)
        xam = np.mean(Xa,axis=1)
        Xap = np.transpose(np.transpose(Xa) - xam)

        if ( do_hybrid ):
            # blend covariance from flow-dependent (ensemble) and static (climatology)
            Bc = (1.0 - hybrid_wght) * Bs + hybrid_wght * B

            # update the central background
            xac, Ac, niters = update_varDA(xbc, Bc, y, R, H, Vupdate=Vupdate, minimization=minimization)

        # write diagnostics to disk before recentering
        if ( do_hybrid ):
            write_diag(diag_fname, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), central_prior=xbc, central_posterior=xac, evratio=evratio, niters=niters)
        else:
            write_diag(diag_fname, k+1, ver, np.transpose(Xb), np.transpose(Xa), y, H, np.diag(R), evratio=evratio)

        # recenter ensemble about central analysis
        if ( do_hybrid ):
            xam = xac.copy()
            if ( hybrid_rcnt ): Xa = np.transpose(xac + np.transpose(Xap))

    print '... all done ...'
    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
