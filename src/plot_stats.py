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
def plot_trace(obs=None, ver=None, xb=None, xa=None, label=['x'], N=1, figNum=None):
    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)
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
def plot_abs_error(xbe, xae, label=['x'], N=1, yscale='semilog', figNum=None):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'

    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)
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
def plot_rmse(xbrmse=None, xarmse=None, xyrmse=None, yscale='semilog', figNum=None, title=None):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'
    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)
    if ( title == None ): title = 'Root Mean Squared Error'

    pyplot.clf()
    pyplot.hold(True)
    if ( yscale == 'linear' ):
        if ( xbrmse != None ): pyplot.plot(xbrmse,'b-',label='prior',      linewidth=2)
        if ( xarmse != None ): pyplot.plot(xarmse,'r-',label='posterior',  linewidth=2)
        if ( xyrmse != None ): pyplot.plot(xyrmse,'k-',label='observation',linewidth=2)
    elif ( yscale == 'semilog' ):
        if ( xbrmse != None ): pyplot.semilogy(xbrmse,'b-',label='prior',      linewidth=2)
        if ( xarmse != None ): pyplot.semilogy(xarmse,'r-',label='posterior',  linewidth=2)
        if ( xyrmse != None ): pyplot.semilogy(xyrmse,'k-',label='observation',linewidth=2)

    yl = pyplot.get(pyplot.gca(),'ylim')
    pyplot.ylim(0.0, yl[1])
    dyl = yl[1]

    if ( xbrmse != None ):
        strb = 'mean prior rmse : %5.4f +/- %5.4f' % (np.mean(xbrmse[100:]), np.std(xbrmse[100:],ddof=1))
        pyplot.text(0.05*len(xbrmse),yl[1]-0.1*dyl,strb,fontsize=10)
    if ( xarmse != None ):
        stra = 'mean posterior rmse : %5.4f +/- %5.4f' % (np.mean(xarmse[100:]), np.std(xarmse[100:],ddof=1))
        pyplot.text(0.05*len(xarmse),yl[1]-0.15*dyl,stra,fontsize=10)
    if ( xyrmse != None ):
        stro = 'mean observation rmse : %5.4f +/- %5.4f' % (np.mean(xyrmse[100:]), np.std(xyrmse[100:],ddof=1))
        pyplot.text(0.05*len(xyrmse),yl[1]-0.2*dyl,stro,fontsize=10)

    pyplot.xlabel('Assimilation Cycle',fontweight='bold',fontsize=12)
    pyplot.ylabel('RMSE',fontweight='bold',fontsize=12)
    pyplot.title(title,fontweight='bold',fontsize=14)
    pyplot.legend(loc=0,ncol=2)
    pyplot.hold(False)
    return fig
###############################################################

###############################################################
def plot_abs_error_var(xbev, xaev, label=['x'], N=1, yscale='semilog', figNum=None):

    if ( (yscale != 'linear') and (yscale != 'semilog') ): yscale = 'semilog'

    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)
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
def plot_iteration_stats(itstats, figNum=None):

    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)

    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(itstats,'k-',linewidth=2)

    yl = pyplot.get(pyplot.gca(),'ylim')
    dyl = yl[1] - yl[0]

    yoff = yl[1] - 0.1 * dyl
    str = 'min  iterations : %d' % (np.min(itstats[1:]))
    pyplot.text(0.05*len(itstats),yoff,str,fontsize=10)
    yoff = yoff - dyl / 20
    str = 'mean iterations : %d' % (np.int(np.mean(itstats[1:])))
    pyplot.text(0.05*len(itstats),yoff,str,fontsize=10)
    yoff = yoff - dyl / 20
    str = 'max  iterations : %d' % (np.max(itstats[1:]))
    pyplot.text(0.05*len(itstats),yoff,str,fontsize=10)
    pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    pyplot.ylabel('# Iterations',     fontweight='bold',fontsize=12)
    pyplot.title('# Iterations for cost function',fontweight='bold',fontsize=14)
    pyplot.hold(False)
    return fig
###############################################################

###############################################################
def plot_error_variance_stats(evratio, figNum=None):

    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)

    pyplot.clf()
    pyplot.hold(True)

    pyplot.plot(evratio,'k-',linewidth=2)
    pyplot.plot(np.ones(len(evratio)-1)*0.5,'r:',linewidth=1)
    pyplot.plot(np.ones(len(evratio)-1)*1.0,'r-',linewidth=1)
    pyplot.plot(np.ones(len(evratio)-1)*2.0,'r:',linewidth=1)
    pyplot.plot(np.ones(len(evratio)-1)*np.mean(evratio[100:]),'g-',linewidth=1)

    pyplot.ylim(0.0,3.0)
    str = 'mean E/V  : %5.4f +/- %5.4f' % (np.mean(evratio[100:]), np.std(evratio[100:],ddof=1))
    pyplot.text(0.05*len(evratio),0.2,str,fontsize=10)

    pyplot.xlabel('Assimilation Step',fontweight='bold',fontsize=12)
    pyplot.ylabel('Innovation Variance / Total Variance',fontweight='bold',fontsize=12)
    pyplot.title('Innovation Variance / Total Variance',fontweight='bold',fontsize=14)
    pyplot.hold(False)

    return fig
###############################################################

###############################################################
def plot_ObImpact(dJa, dJe, figNum=None, sOI=0, eOI=None):

    if ( figNum == None ): fig = pyplot.figure()
    else: fig = pyplot.figure(figNum)
    pyplot.clf()
    pyplot.hold(True)

    color_adj = 'b'
    color_ens = 'r'

    if eOI == None: eOI = len(dJa)
    if eOI == -1:   eOI = len(dJa)
    index = np.arange(eOI-sOI)
    width = 0.49
    zeroline = np.zeros(len(dJa[sOI:eOI+1]))

    ra = pyplot.bar(index, dJa[sOI:eOI], width, color=color_adj, edgecolor=color_adj, label='ASA')
    stra = r'mean $\delta J_a$ : %5.4f +/- %5.4f' % (np.mean(dJa), np.std(dJa,ddof=1))

    re = pyplot.bar(index+width, dJe[sOI:eOI], width, color=color_ens, edgecolor=color_ens, label='ESA')
    stre = r'mean $\delta J_e$ : %5.4f +/- %5.4f' % (np.mean(dJe), np.std(dJe,ddof=1))

    pyplot.plot(zeroline,'k-',linewidth=1)

    if   ( len(zeroline) >= 1000 ):
        inc = 1000
    elif ( len(zeroline) >= 100 ):
        inc = 100
    elif ( len(zeroline) >= 10 ):
        inc = 10
    else:
        inc = 1

    locs, labels = pyplot.xticks()
    newlocs   = np.arange(sOI,sOI+len(zeroline)+1,inc) - sOI
    newlabels = np.arange(sOI,sOI+len(zeroline)+1,inc)
    pyplot.xticks(newlocs, newlabels)

    yl = pyplot.get(pyplot.gca(),'ylim')
    dyl = yl[1] - yl[0]
    yoff = yl[0] + 0.1 * dyl
    pyplot.text(5,yoff,stra,fontsize=10,color=color_adj)
    yoff = yl[0] + 0.2 * dyl
    pyplot.text(5,yoff,stre,fontsize=10,color=color_ens)
    pyplot.xlim(newlabels[0], len(zeroline))

    pyplot.xlabel('Assimilation Step', fontweight='bold',fontsize=12)
    pyplot.ylabel(r'$\delta J$',       fontweight='bold',fontsize=12)
    pyplot.title('Observation Impact',fontweight='bold',fontsize=14)
    pyplot.legend(loc=0,ncol=2)
    pyplot.hold(False)
    return fig
###############################################################
