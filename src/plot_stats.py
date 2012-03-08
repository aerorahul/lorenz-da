#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# plot_stats.py - Functions related to plotting statistics
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import numpy      as     np
from   matplotlib import pyplot
###############################################################

###############################################################
def plot_trace(obs=None, ver=None, xb=None, xa=None, label=['x'], N=1):
    fig = pyplot.figure()
    pyplot.clf()
    for k in range(0,N):
        pyplot.subplot(N,1,k+1)
        pyplot.hold(True)
        if ( obs != None ): pyplot.plot(obs[k,:],'ro',label='observation')
        if ( ver != None ): pyplot.plot(ver[k,:],'k-',label='truth')
        if ( xb  != None ): pyplot.plot(xb[k,:], 'c-',label='background')
        if ( xa  != None ): pyplot.plot(xa[k,:], 'b-',label='analysis')
        pyplot.ylabel(label[k],fontweight='bold',fontsize=12)
        if ( label[k] != 'z' ): pyplot.plot(np.zeros(len(ver[k,:])),'k:')
        pyplot.hold(False)
        if ( k == 0 ):
            pyplot.title('Time trace',fontweight='bold',fontsize=14)
        if ( k == 1 ):
            pyplot.legend(loc=0,ncol=2)
        if ( k == N-1 ):
            pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    return fig
###############################################################

###############################################################
def plot_abs_error(xbe, xae, label=['x'], N=1, yscale='semilog'):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    for k in range(0,N):
        pyplot.subplot(N,1,k+1)
        if ( yscale == 'linear' ):
            pyplot.plot(np.abs(xbe[k,:]),'b-',label='background',linewidth=2)
            pyplot.plot(np.abs(xae[k,:]),'r-',label='analysis',linewidth=2)
        elif ( yscale == 'semilog' ):
            pyplot.semilogy(np.abs(xbe[k,:]),'b-',label='background',linewidth=2)
            pyplot.semilogy(np.abs(xae[k,:]),'r-',label='analysis',linewidth=2)
        pyplot.ylabel(label[k],fontweight='bold',fontsize=12)
        strb = 'mean background : %5.4f +/- %5.4f' % (np.mean(np.abs(xbe[k,1:])), np.std(np.abs(xbe[k,1:]),ddof=1))
        stra = 'mean analysis : %5.4f +/- %5.4f' % (np.mean(np.abs(xae[k,1:])), np.std(np.abs(xae[k,1:]),ddof=1))
        yl = pyplot.get(pyplot.gca(),'ylim')
        if ( yscale == 'linear' ):
            yoffb = 0.9 * yl[1]
            yoffa = 0.8 * yl[1]
        elif ( yscale == 'semilog' ):
            yoffb = 0.5e1 * yl[0]
            yoffa = 0.2e1 * yl[0]
        pyplot.text(0,yoffb,strb,fontsize=10)
        pyplot.text(0,yoffa,stra,fontsize=10)
        if ( k == 0 ):
            pyplot.title('Absolute Error',fontweight='bold',fontsize=14)
        if ( k == 1 ):
            pyplot.legend(loc=0)
        if ( k == N-1 ):
            pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    return fig
###############################################################

###############################################################
def plot_rmse(xbrmse, xarmse, xyrmse=None, yscale='semilog'):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    if ( yscale == 'linear' ):
        pyplot.plot(xbrmse,'b-',label='prior',    linewidth=2)
        pyplot.plot(xarmse,'r-',label='posterior',linewidth=2)
        if ( xyrmse != None ):
            pyplot.plot(xyrmse,'k-',label='observation',linewidth=2)
    elif ( yscale == 'semilog' ):
        pyplot.semilogy(xbrmse,'b-',label='prior',    linewidth=2)
        pyplot.semilogy(xarmse,'r-',label='posterior',linewidth=2)
        if ( xyrmse != None ):
            pyplot.semilogy(xyrmse,'k-',label='observation',linewidth=2)
    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title('Root Mean Squared Error',fontweight='bold',fontsize=14)
    pyplot.legend(loc=0)
    pyplot.hold(False)
    return fig
###############################################################

###############################################################
def plot_abs_error_var(xbev, xaev, label=['x'], N=1, yscale='semilog'):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'

    fig = pyplot.figure()
    pyplot.clf()
    for k in range(0,N):
        pyplot.subplot(N,1,k+1)
        pyplot.hold(True)
        if ( yscale == 'linear'):
            pyplot.plot(xbev[k,2:],'b-',label='background',linewidth=2)
            pyplot.plot(xaev[k,2:],'r-',label='analysis',linewidth=2)
        elif ( yscale == 'semilog' ):
            pyplot.semilogy(np.abs(xbev[k,2:]),'b-',label='background',linewidth=2)
            pyplot.semilogy(np.abs(xaev[k,2:]),'r-',label='analysis',linewidth=2)
        pyplot.ylabel(label[k],fontweight='bold',fontsize=12)
        strb = 'mean background : %5.4f +/- %5.4f' % (np.mean(xbev[k,2:]), np.std(xbev[k,2:],ddof=1))
        stra = 'mean analysis : %5.4f +/- %5.4f' % (np.mean(xaev[k,2:]), np.std(xaev[k,2:],ddof=1))
        yl = pyplot.get(pyplot.gca(),'ylim')
        if ( yscale == 'linear'):
            yoffb = 0.9 * yl[1]
            yoffa = 0.8 * yl[1]
        elif ( yscale == 'semilog' ):
            yoffb = 0.5e1 * yl[0]
            yoffa = 0.2e1 * yl[0]
        pyplot.text(0,yoffb,strb,fontsize=10)
        pyplot.text(0,yoffa,stra,fontsize=10)
        pyplot.hold(False)
        if ( k == 0 ):
            pyplot.title('Ensemble Kalman Filter Error Variance ',fontweight='bold',fontsize=14)
        if ( k == 1 ):
            pyplot.legend(loc=1)
        if ( k == N-1 ):
            pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    return fig
###############################################################

###############################################################
def plot_iteration_stats(itstats):
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(itstats,'k-',linewidth=2)
    yl = pyplot.get(pyplot.gca(),'ylim')
    yoff = yl[0] + 0.25 * (yl[1] - yl[0])
    str = 'min  iterations : %d' % (np.min(itstats))
    pyplot.text(2,yoff,str,fontsize=10)
    yoff = yoff - 2
    str = 'mean iterations : %d' % (np.int(np.mean(itstats)))
    pyplot.text(2,yoff,str,fontsize=10)
    yoff = yoff - 2
    str = 'max  iterations : %d' % (np.max(itstats))
    pyplot.text(2,yoff,str,fontsize=10)
    pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    pyplot.ylabel('# Iterations',     fontweight='bold',fontsize=12)
    pyplot.title('# Iterations for cost function',fontweight='bold',fontsize=14)
    pyplot.hold(False)
    return fig
###############################################################

###############################################################
def plot_error_variance_stats(evratio):
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(evratio,'k-',linewidth=2)
    pyplot.plot(np.ones(len(evratio))*0.5,'r:')
    pyplot.plot(np.ones(len(evratio))*1.0,'k-')
    pyplot.plot(np.ones(len(evratio))*2.0,'r:')
    pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    pyplot.ylabel('Innovation Variance / Total Variance',fontweight='bold',fontsize=12)
    pyplot.title('Innovation Variance / Total Variance',fontweight='bold',fontsize=14)
    pyplot.hold(False)
    return fig
###############################################################

###############################################################
def plot_ObImpact(dJa=None, dJe=None):
    if ( (dJa == None) and (dJe == None) ):
        print 'dJa == dJe == None, nothing to plot'
        return None
    if ( (dJa != None) and (dJe != None) ): width = 0.5
    if ( (dJa != None) or  (dJe != None) ): width = 1.0
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    yl = pyplot.get(pyplot.gca(),'ylim')
    if ( dJa != None ):
        pyplot.plot(dJa,'bo-',label='Adjoint', linewidth=2)
        pyplot.plot(np.ones(len(dJa))*np.mean(dJa),'b:')
        yoff = yl[0] + 15
        stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(dJa), np.std(dJa,ddof=1))
        pyplot.text(1,yoff,stra,fontsize=10)
        zeroline = np.zeros(len(dJa))
    if ( dJe != None ):
        pyplot.plot(dJe,'ro-',label='Ensemble',linewidth=2)
        pyplot.plot(np.ones(len(dJe))*np.mean(dJe),'r:')
        stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(dJe), np.std(dJe,ddof=1))
        yoff = yoff + 10
        pyplot.text(1,yoff,stre,fontsize=10)
        zeroline = np.zeros(len(dJe))
    pyplot.plot(zeroline,'k:')
    pyplot.xlabel('Assimilation Step', fontweight='bold',fontsize=12)
    pyplot.ylabel('delta J',           fontweight='bold',fontsize=12)
    pyplot.title('Observation Impact',fontweight='bold',fontsize=14)
    pyplot.legend(loc=4,ncol=2)
    pyplot.hold(False)
    return fig
###############################################################
