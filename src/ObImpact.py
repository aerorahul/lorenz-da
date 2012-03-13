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
from   plot_stats    import *
###############################################################

###############################################################
def main():

    # model
    model = 'L96'

    # Ob. Impact parameters
    nf  =  4         # length of the extended forecast
    sOI = 501        # start doing ob. impact at sOI
    eOI = -1         # stop  doing ob. impact at eOI

    # name of output diagnostic file
    dir_data       = '../data/' + model + '/ensDA_N=40/inf=1.21/'
    fname_diag     = dir_data + model + '_ensDA_diag.nc4'
    fname_ObImpact = dir_data + model + '_ensDA_ObImpact'

    # read dimensions and necessary attributes from the diagnostic file
    try:
        nc = Dataset(fname_diag, mode='r', format='NETCDF4')
        ndof   = len(nc.dimensions['ndof'])
        nassim = len(nc.dimensions['ntime'])
        nobs   = len(nc.dimensions['nobs'])

        if 'ncopy' in nc.dimensions:
            nens = len(nc.dimensions['ncopy'])
        else:
            nens = 0

        if 'do_hybrid' in nc.ncattrs():
            do_hybrid = nc.do_hybrid
        else:
            do_hybrid = False

        ntimes = nc.ntimes
        dt     = nc.dt
        F      = nc.F
        dF     = nc.dF

        nc.close()
    except Exception as Instance:
        print 'Exception occurred during read of ' + fname_diag
        print type(Instance)
        print Instance.args
        print Instance
        sys.exit(1)

    tf = np.arange(0.0,nf*ntimes+dt,dt)  # extended forecast

    mxf    = np.zeros(ndof)
    mxf[0] = 1.0                    # metric: single variable
    mxf    = np.ones(ndof)          # metric: sum of variables

    # allocate appropriate space for variables
    if ( sOI < 0 ): sOI = 1
    if ( eOI < 0 ): eOI = nassim
    e_dJb = np.zeros(eOI - sOI) * np.NaN
    e_dJa = np.zeros(eOI - sOI) * np.NaN
    e_dJ  = np.zeros(eOI - sOI) * np.NaN
    a_dJb = np.zeros(eOI - sOI) * np.NaN
    a_dJa = np.zeros(eOI - sOI) * np.NaN
    a_dJ  = np.zeros(eOI - sOI) * np.NaN

    # read diagnostics from file
    for k in range(sOI,eOI):

        print '========== assimilation time = %d ========== ' % (k)

        if ( do_hybrid ):
            xti, Xbi, Xai, y, H, R, xbmi, xami, niter, evratio = read_diag(fname_diag, k)
        else:
            if ( nens == 0 ):
                xti, Xbi, Xai, y, H, R, niters  = read_diag(fname_diag, k)
            else:
                xti, Xbi, Xai, y, H, R, evratio = read_diag(fname_diag, k)

        # transpose required because of the way data is written to disk
        Xbi = np.transpose(Xbi)
        Xai = np.transpose(Xai)

        if ( nens != 0 ):
            if ( not do_hybrid ):
                xbmi = np.mean(Xbi,axis=1)
                xami = np.mean(Xai,axis=1)
            Xbpi = np.transpose(np.transpose(Xbi) - xbmi)
            Xapi = np.transpose(np.transpose(Xai) - xami)

        # construct covariances
        B = np.dot(Xbpi,np.transpose(Xbpi)) / (nens - 1)
        A = np.dot(Xapi,np.transpose(Xapi)) / (nens - 1)

        # compute innovation
        ye  = np.dot(H,Xbi)
        mye = np.mean(ye,axis=1)
        dy  = y - mye

        # advance truth
        exec('xf = integrate.odeint(%s, xti, tf, (F,0.0))' % model)
        xtf = xf[-1,:].copy()

        # advance background
        Xbf = np.zeros((ndof,nens))
        for m in range(0,nens):
            xb = Xbi[:,m].copy()
            exec('xf = integrate.odeint(%s, xb, tf, (F+dF,0.0))' % model)
            Xbf[:,m] = xf[-1,:].copy()

        # advance analysis
        Xaf = np.zeros((ndof,nens))
        for m in range(0,nens):
            xa = Xai[:,m].copy()
            exec('xf = integrate.odeint(%s, xa, tf, (F+dF,0.0))' % model)
            Xaf[:,m] = xf[-1,:].copy()

        # advance background and analysis mean
        exec('xbmf = integrate.odeint(%s, xbmi, tf, (F+dF,0.0))' % model)
        exec('xamf = integrate.odeint(%s, xami, tf, (F+dF,0.0))' % model)

        index = k - sOI

        # metric : J = (x^T)Wx ; dJ/dx = J_x = 2Wx ; choose W = I, x = xfmet, J_x = Jxf

        xfmet = np.transpose(mxf * (np.transpose(Xbf) - xtf))
        Jb = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jbp = Jb - np.mean(Jb,axis=0)
        HXbp = np.dot(H,Xbpi)
        Kmb = np.linalg.inv(np.cov(np.dot(H,Xbi),ddof=1) + R)
        JbHXb = np.dot(Jbp,np.transpose(HXbp)) / (nens - 1)
        e_dJb[index] = np.dot(JbHXb,np.dot(Kmb,dy))

        xfmet = np.transpose(mxf * (np.transpose(Xaf) - xtf))
        Ja = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jap = Ja - np.mean(Ja,axis=0)
        HXap = np.dot(H,Xapi)
        Kma = np.linalg.inv(R)
        JaHXa = np.dot(Jap,np.transpose(HXap)) / (nens - 1)
        e_dJa[index] = np.dot(JaHXa,np.dot(Kma,dy))

        e_dJ[index] = e_dJb[index] + e_dJa[index]

        print 'dJe = %5.4f | dJe_a = %5.4f | dJe_b = %5.4f ' % (e_dJ[index], e_dJa[index], e_dJb[index] )

        Jxbf = mxf * (xbmf[-1,:] - xtf)
        exec('Jxb  = integrate.odeint(%s_tlm, Jxbf, tf, (F+dF,np.flipud(xbmf),tf,True))' % model)
        Jxbi = Jxb[-1,:].copy()
        a_dJb[index] = np.dot(Jxbi,np.dot(B,np.dot(np.transpose(H),np.dot(np.linalg.inv(np.cov(np.dot(H,Xbi)) + R),dy))))

        Jxaf = mxf * (xamf[-1,:] - xtf)
        exec('Jxa  = integrate.odeint(%s_tlm, Jxaf, tf, (F+dF,np.flipud(xamf),tf,True))' % model)
        Jxai = Jxa[-1,:].copy()
        a_dJa[index] = np.dot(Jxai,np.dot(A,np.dot(np.transpose(H),np.dot(np.linalg.inv(R),dy))))

        a_dJ[index] = a_dJb[index] + a_dJa[index]

        print 'dJa = %5.4f | dJa_a = %5.4f | dJa_b = %5.4f ' % (a_dJ[index], a_dJa[index], a_dJb[index] )

    # write the Ob. Impact to disk
    object = {'ens_dJ' : e_dJ, 'ens_dJb' : e_dJb, 'ens_dJa' : e_dJa,
              'adj_dJ' : a_dJ, 'adj_dJb' : a_dJb, 'adj_dJa' : a_dJa}
    fh = open(fname_ObImpact+'.dat','wb')
    cPickle.dump(object,fh,2)
    fh.close()

    fig = plot_ObImpact(dJa=a_dJ, dJe=e_dJ, startxIndex=sOI)
    fig.savefig(fname_ObImpact+'.png',dpi=100,orientation='landscape',format='png')
    fig.savefig(fname_ObImpact+'.eps',dpi=300,orientation='landscape',format='eps')
    pyplot.show()
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
