#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# ObImpact.py - compute observation impact from Adjoint- and
#               Ensemble- based sensitivity analysis
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
import cPickle       as     cPickle
from   matplotlib    import pyplot
from   netCDF4       import Dataset
from   scipy         import integrate, io
from   module_Lorenz import *
from   module_IO     import *
###############################################################

###############################################################
def main():

    # read in input arguments
    model = str(sys.argv[1])
    nf    = int(sys.argv[2])
    sOI   = int(sys.argv[3])
    eOI   = int(sys.argv[4])
    fname = str(sys.argv[5])

    fname_ObImpact = fname.replace('.nc4','.dat')
    fname_ObImpact = fname_ObImpact.replace('diag','ObImpact')

    # read dimensions and necessary attributes from the diagnostic file
    try:
        nc = Dataset(fname, mode='r', format='NETCDF4')
        ndof   = len(nc.dimensions['ndof'])
        nassim = len(nc.dimensions['ntime'])
        nobs   = len(nc.dimensions['nobs'])
        nens   = len(nc.dimensions['ncopy'])

        if 'do_hybrid' in nc.ncattrs():
            do_hybrid   = nc.do_hybrid
            hybrid_wght = nc.hybrid_wght
        else:
            do_hybrid = False

        ntimes = nc.ntimes
        dt     = nc.dt
        F      = nc.F
        dF     = nc.dF

        nc.close()
    except Exception as Instance:
        print 'Exception occurred during read of ' + fname
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    # load climatological covariance
    if ( do_hybrid ):
        print 'load climatological covariance ...'
        nc = Dataset('L96_climo_B.nc4','r')
        Bc = nc.variables['B'][:]
        nc.close()

    tf = np.arange(0.0,nf*ntimes+dt,dt)  # extended forecast

    mxf    = np.zeros(ndof)
    mxf[0] = 1.0                    # metric: single variable
    mxf    = np.ones(ndof)          # metric: sum of variables

    if ( sOI < 0 ): sOI = 1
    if ( eOI < 0 ): eOI = nassim

    # allocate appropriate space for variables
    e_dJb = np.zeros(eOI - sOI) * np.NaN
    e_dJa = np.zeros(eOI - sOI) * np.NaN
    e_dJ  = np.zeros(eOI - sOI) * np.NaN
    a_dJb = np.zeros(eOI - sOI) * np.NaN
    a_dJa = np.zeros(eOI - sOI) * np.NaN
    a_dJ  = np.zeros(eOI - sOI) * np.NaN

    # read diagnostics from file
    for k in range(sOI,eOI):

        print '========== assimilation time = %5d ========== ' % (k)

        if ( do_hybrid ):
            xti, Xbi, Xai, y, H, R, xbci, xaci, tmp, tmp = read_diag(fname, k)
        else:
            xti, Xbi, Xai, y, H, R, tmp  = read_diag(fname, k)

        # transpose required because of the way data is written to disk
        Xbi = np.transpose(Xbi)
        Xai = np.transpose(Xai)

        # compute ensemble mean and perturbations
        xbmi = np.mean(Xbi,axis=1)
        xami = np.mean(Xai,axis=1)
        Xbpi = np.transpose(np.transpose(Xbi) - xbmi)
        Xapi = np.transpose(np.transpose(Xai) - xami)

        # construct covariances
        B = np.cov(Xbi,ddof=1)
        if ( do_hybrid ): Bs = (1 - hybrid_wght) * Bc + hybrid_wght * B

        # compute the Kalman gain
        if ( do_hybrid ):
            K = np.dot(Bs,np.dot(np.transpose(H),np.linalg.inv(np.dot(H,np.dot(Bs,np.transpose(H))) + R)))
        else:
            K = np.dot(B,np.dot(np.transpose(H),np.linalg.inv(np.cov(np.dot(H,Xbi),ddof=1) + R)))

        # compute innovation
        if ( do_hybrid ): ye = np.dot(H,xbci)
        else:             ye = np.dot(H,xbmi)
        dy  = y - ye

        # advance truth
        exec('xf = integrate.odeint(%s, xti, tf, (F,0.0))' % model)
        xtf = xf[-1,:].copy()

        # advance background ensemble
        Xbf = np.zeros((ndof,nens))
        for m in range(0,nens):
            xb = Xbi[:,m].copy()
            exec('xf = integrate.odeint(%s, xb, tf, (F+dF,0.0))' % model)
            Xbf[:,m] = xf[-1,:].copy()

        # advance analysis ensemble
        Xaf = np.zeros((ndof,nens))
        for m in range(0,nens):
            if ( do_hybrid ):
                xa = np.squeeze(Xai[:,m]) - xami + xaci
            else:
                xa = Xai[:,m].copy()
            exec('xf = integrate.odeint(%s, xa, tf, (F+dF,0.0))' % model)
            Xaf[:,m] = xf[-1,:].copy()

        # advance background and analysis mean / central
        if ( do_hybrid ):
            exec('xbmf = integrate.odeint(%s, xbci, tf, (F+dF,0.0))' % model)
            exec('xamf = integrate.odeint(%s, xaci, tf, (F+dF,0.0))' % model)
        else:
            exec('xbmf = integrate.odeint(%s, xbmi, tf, (F+dF,0.0))' % model)
            exec('xamf = integrate.odeint(%s, xami, tf, (F+dF,0.0))' % model)

        index = k - sOI

        # metric : J = (x^T)Wx ; dJ/dx = J_x = 2Wx ; choose W = I, x = xfmet, J_x = Jxf

        # ensemble-based observation impact

        xfmet = np.transpose(mxf * (np.transpose(Xbf) - xtf))
        Jb = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jbp = Jb - np.mean(Jb,axis=0)
        JbHXb = np.dot(Jbp,np.transpose(np.dot(H,Xbpi))) / (nens - 1)
        Kmb = np.linalg.inv(np.cov(np.dot(H,Xbi),ddof=1) + R)
        e_dJb[index] = np.dot(JbHXb,np.dot(Kmb,dy))

        xfmet = np.transpose(mxf * (np.transpose(Xaf) - xtf))
        Ja = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jap = Ja - np.mean(Ja,axis=0)
        JaHXa = np.dot(Jap,np.transpose(np.dot(H,Xapi))) / (nens - 1)
        Kma = np.linalg.inv(R)
        e_dJa[index] = np.dot(JaHXa,np.dot(Kma,dy))

        e_dJ[index] = e_dJb[index] + e_dJa[index]

        print 'dJe = %5.4f | dJe_a = %5.4f | dJe_b = %5.4f ' % (e_dJ[index], e_dJa[index], e_dJb[index] )

        # adjoint-based observation impact

        Jxbf = mxf * (xbmf[-1,:] - xtf)
        exec('Jxb  = integrate.odeint(%s_tlm, Jxbf, tf, (F+dF,np.flipud(xbmf),tf,True))' % model)
        Jxbi = Jxb[-1,:].copy()
        a_dJb[index] = np.dot(Jxbi,np.dot(K,dy))

        Jxaf = mxf * (xamf[-1,:] - xtf)
        exec('Jxa  = integrate.odeint(%s_tlm, Jxaf, tf, (F+dF,np.flipud(xamf),tf,True))' % model)
        Jxai = Jxa[-1,:].copy()
        a_dJa[index] = np.dot(Jxai,np.dot(K,dy))

        a_dJ[index] = a_dJb[index] + a_dJa[index]

        print 'dJa = %5.4f | dJa_a = %5.4f | dJa_b = %5.4f ' % (a_dJ[index], a_dJa[index], a_dJb[index] )

    # write the Ob. Impact to disk
    object = {'ens_dJ' : e_dJ, 'ens_dJb' : e_dJb, 'ens_dJa' : e_dJa,
              'adj_dJ' : a_dJ, 'adj_dJb' : a_dJb, 'adj_dJa' : a_dJa}
    fh = open(fname_ObImpact, 'wb')
    cPickle.dump(object,fh,2)
    fh.close()

    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
