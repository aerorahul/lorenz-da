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
import numpy          as     np
import cPickle        as     cPickle
from   matplotlib     import pyplot
from   netCDF4        import Dataset
from   module_Lorenz  import *
from   module_IO      import *
###############################################################

###############################################################
def main():

    # length of the extended forecast (in multiples of assimilation length):
    nf = 4

    # read in input arguments:
    [_, fname, sOI, eOI] = get_input_arguments()

    # get model, DA class data and necessary attributes from the diagnostic file:
    [model, DA, ensDA, varDA] = read_diag_info(fname)

    # choose metric:
    mxf    = np.zeros(model.Ndof)
    mxf[0] = 1.0                    # metric: single variable
    mxf    = np.ones(model.Ndof)    # metric: sum of variables

    # check for starting and ending indices
    if ( sOI < 0 ): sOI = 1
    if ( eOI < 0 ): eOI = DA.nassim

    # allocate appropriate space for variables upfront:
    e_dJb = np.zeros(((eOI - sOI), model.Ndof)) * np.NaN
    e_dJa = np.zeros(((eOI - sOI), model.Ndof)) * np.NaN
    a_dJb = np.zeros(((eOI - sOI), model.Ndof)) * np.NaN
    a_dJa = np.zeros(((eOI - sOI), model.Ndof)) * np.NaN

    # time-vector for DA.t0 to nf*DA.ntimes:
    tf = np.arange(DA.t0,nf*DA.ntimes+model.dt,model.dt)

    # load climatological covariance once and for all:
    if ( DA.do_hybrid ):
        print 'load climatological covariance for %s ...' % (model.Name)
        nc = Dataset(model.Name + '_climo_B.nc4','r')
        Bc = nc.variables['B'][:]
        nc.close()

    for k in range(sOI,eOI):

        print '========== assimilation time = %5d ========== ' % (k)

        # read diagnostics from file
        if ( DA.do_hybrid ):
            xti, Xbi, Xai, y, H, R, xbci, xaci, _, _ = read_diag(fname, k)
        else:
            xti, Xbi, Xai, y, H, R, _                = read_diag(fname, k)

        # This is only applicable, when H and R are stored as (Ndof,Ndof) matrices in diag files.
        # Newer versions save H and R and vectors of length (Ndof)
        # It will eventually be phased out.
        if ( len(np.shape(H)) == 1 ): H = np.diag(H)
        if ( len(np.shape(R)) == 1 ): R = np.diag(R)

        # if Nobs < Ndof, find the obs. index that is valid
        valInd = np.isfinite(y)

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
        if ( DA.do_hybrid ): Bs = (1.0 - DA.hybrid_wght) * Bc + DA.hybrid_wght * B

        # compute innovation
        if ( DA.do_hybrid ): ye = np.dot(H,xbci)
        else:                ye = np.dot(H,xbmi)
        dy = y[valInd] - ye[valInd]

        # advance truth
        xf = advance_model(model, xti, tf, perfect=True)
        xtf = xf[-1,:].copy()

        # advance background ensemble
        Xbf = np.zeros((model.Ndof,ensDA.Nens))
        for m in range(0,ensDA.Nens):
            xb = Xbi[:,m].copy()
            xf = advance_model(model, xb, tf, perfect=False)
            Xbf[:,m] = xf[-1,:].copy()

        # advance analysis ensemble
        Xaf = np.zeros((model.Ndof,ensDA.Nens))
        for m in range(0,ensDA.Nens):
            if ( DA.do_hybrid ):
                if ( DA.hybrid_rcnt ): xa = np.squeeze(Xai[:,m]) - xami + xaci
                else:                  xa = Xai[:,m].copy()
            else:
                xa = Xai[:,m].copy()
            xf = advance_model(model, xa, tf, perfect=False)
            Xaf[:,m] = xf[-1,:].copy()

        # advance background and analysis mean / central
        if ( DA.do_hybrid ):
            xbmf = advance_model(model, xbci, tf, perfect=False)
            xamf = advance_model(model, xaci, tf, perfect=False)
        else:
            xbmf = advance_model(model, xbmi, tf,  perfect=False)
            xamf = advance_model(model, xami, tf,  perfect=False)

        index = k - sOI

        # metric : J = (x^T)Wx ; dJ/dx = J_x = 2Wx ; choose W = I, x = xfmet, J_x = Jxf

        # E N S E M B L E - based observation impact

        xfmet = np.transpose(mxf * (np.transpose(Xbf) - xtf))
        Jb = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jbp = Jb - np.mean(Jb,axis=0)
        JbHXb = np.dot(Jbp,np.transpose(np.dot(H[valInd,:],Xbpi))) / (ensDA.Nens - 1)
        Kmb = np.linalg.inv(np.cov(np.dot(H[valInd,:],Xbi),ddof=1) + np.diag(R[valInd,valInd]))

        xfmet = np.transpose(mxf * (np.transpose(Xaf) - xtf))
        Ja = 0.5 * np.diag(np.dot(np.transpose(xfmet), xfmet))
        Jap = Ja - np.mean(Ja,axis=0)
        JaHXa = np.dot(Jap,np.transpose(np.dot(H[valInd,:],Xapi))) / (ensDA.Nens - 1)
        Kma = np.linalg.inv(np.diag(R[valInd,valInd]))

        e_dJb[index,valInd] = JbHXb * np.dot(Kmb,dy)
        e_dJa[index,valInd] = JaHXa * np.dot(Kma,dy)

        print 'dJe = %12.5f | dJe_a = %12.5f | dJe_b = %12.5f ' % ( np.nansum(e_dJa[index,:] + e_dJb[index,:]), np.nansum(e_dJa[index,:]), np.nansum(e_dJb[index,:]) )

        # A D J O I N T - based observation impact

        if ( DA.do_hybrid ):
            K = np.dot(Bs,np.dot(np.transpose(H[valInd,:]),np.linalg.inv(np.dot(H[valInd,:],np.dot(Bs,np.transpose(H[valInd,:]))) + np.diag(R[valInd,valInd]))))
        else:
            K = np.dot(B,np.dot(np.transpose(H[valInd,:]),np.linalg.inv(np.cov(np.dot(H[valInd,:],Xbi),ddof=1) + np.diag(R[valInd,valInd]))))

        Jxbf = mxf * (xbmf[-1,:] - xtf)
        Jxb  = advance_model_tlm(model, Jxbf, tf, xbmf, tf, adjoint=True, perfect=False)
        Jxbi = Jxb[-1,:].copy()

        Jxaf = mxf * (xamf[-1,:] - xtf)
        Jxa  = advance_model_tlm(model, Jxaf, tf, xamf, tf, adjoint=True, perfect=False)
        Jxai = Jxa[-1,:].copy()

        a_dJb[index,:] = Jxbi * np.dot(K,dy)
        a_dJa[index,:] = Jxai * np.dot(K,dy)

        print 'dJa = %12.5f | dJa_a = %12.5f | dJa_b = %12.5f ' % (np.nansum(a_dJa[index,:]+a_dJb[index,:]), np.nansum(a_dJa[index,:]), np.nansum(a_dJb[index,:]) )

    # write the observation impact to disk:
    object = {'ens_dJb' : e_dJb, 'ens_dJa' : e_dJa,
              'adj_dJb' : a_dJb, 'adj_dJa' : a_dJa}

    fname_ObImpact = fname.replace('.nc4','.dat')
    fname_ObImpact = fname_ObImpact.replace('diag','ObImpact')

    fh = open(fname_ObImpact, 'wb')
    cPickle.dump(object,fh,2)
    fh.close()

    sys.exit(0)
###############################################################

###############################################################
if __name__ == "__main__":
	main()
###############################################################
