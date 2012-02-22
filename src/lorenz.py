#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# lorenz.py - Functions related to the Lorenz model
###############################################################

__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"

import numpy as np
from matplotlib import pyplot

def L63(x0, t, par, dummy):
# {{{
    '''
    L63 - function that integrates the Lorenz 1963 equations, given parameters 'par' and initial
    conditions 'x0'

    xs = L63(x0, t, (par, dummy))

       xs - final state at time t = T
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
      par - parameters of the Lorenz 1963 system
    dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
    '''

    x, y, z = x0
    s, r, b = par

    x_dot = s*(y-x)
    y_dot = r*x -y -x*z
    z_dot = x*y - b*z

    xs = np.array([x_dot, y_dot, z_dot])

    return xs
# }}}

def L63_tlm(x0, t, par, xsave, tsave, adjoint):
# {{{
    '''
    L63_tlm - function that integrates the Lorenz 1963 equations forward or backward using a TLM and
    its adjoint, given parameters 'par' and initial perturbations 'x0'

    xs = L63_tlm(x0, t, (par, xsave, tsave, adjoint))

         xs - evolved perturbations at time t = T
         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
        par - parameters of the Lorenz 1963 system
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
    '''

    s, r, b = par

    x = np.interp(t,tsave,xsave[:,0])
    y = np.interp(t,tsave,xsave[:,1])
    z = np.interp(t,tsave,xsave[:,2])

    M = np.array([[-s,   s,  0],
                  [r-z, -1, -x],
                  [y,    x, -b]])

    if ( adjoint ):
        xs = np.dot(np.transpose(M),x0)
    else:
        xs = np.dot(M,x0)

    return xs
# }}}

def L96(x0, t, F, dummy):
# {{{
    '''
    L96 - function that integrates the Lorenz and Emanuel 1998 equations, given forcing 'F' and initial
    conditions 'x0'

    xs = L96(x0, t, (F, dummy))

       xs - final state at time t = T
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
        F - Forcing
    dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
    '''

    Ndof = len(x0)
    xs = np.zeros(Ndof)

    for j in range(0,Ndof):
        jp1 = j + 1
        if ( jp1 >= Ndof ): jp1 = jp1 - Ndof
        jm2 = j - 2
        if ( jm2 < 0 ): jm2 = Ndof + jm2
        jm1 = j - 1
        if ( jm1 < 0 ): jm1 = Ndof + jm1

        xs[j] = ( x0[jp1] - x0[jm2] ) * x0[jm1] - x0[j] + F

    return xs
# }}}

def L96_tlm(x0, t, F, xsave, tsave, adjoint):
# {{{
    '''
    L96_tlm - function that integrates the Lorenz and Emanuel 1998 equations forward or backward
    using a TLM and its adjoint, given Forcing 'F' and initial perturbations 'x0'

    xs = L96_tlm(x0, t, (F, xsave, tsave, adjoint))

         xs - evolved perturbations at time t = T
         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
          F - Forcing
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
    '''

    Ndof = len(x0)
    x = np.zeros(Ndof)

    for j in range(0,Ndof):
        x[j] = np.interp(t,tsave,xsave[:,j])

    M = np.zeros((Ndof,Ndof))

    for j in range(0,Ndof):
        jp1 = j + 1
        if ( jp1 >= Ndof ): jp1 = jp1 - Ndof
        jm2 = j - 2
        if ( jm2 < 0 ): jm2 = Ndof + jm2
        jm1 = j - 1
        if ( jm1 < 0 ): jm1 = Ndof + jm1

        M[j,jm2] = -x[jm1]
        M[j,jm1] = x[jp1] - x[jm2]
        M[j,j]   = -1
        M[j,jp1] = x[jm1]

    if ( adjoint ):
        xs = np.dot(np.transpose(M),x0)
    else:
        xs = np.dot(M,x0)

    return xs
# }}}

def plot_L63(attractor,xdim=0,ydim=2,segment=None):
# {{{
    '''
    Plot the Lorenz 1963 attractor in 2D

    plot_L63(attractor, xdim=0, ydim=2, segment=None)

    attractor - x,y,z from t = [0, T]
         xdim - variable along x-axis (X)
         ydim - variable along y-axis (Z)
      segment - overlay segment on attractor (None)
    '''

    if ( xdim == ydim ):
        xdim = 0
        ydim = 2

    if ( xdim < 0  or xdim > 2 ) : xdim = 0
    if ( ydim < 0  or ydim > 2 ) : ydim = 2

    if   ( xdim == 0 ): xlab = 'X'
    elif ( xdim == 1 ): xlab = 'Y'
    elif ( xdim == 2 ): xlab = 'Z'

    if   ( ydim == 0 ): ylab = 'X'
    elif ( ydim == 1 ): ylab = 'Y'
    elif ( ydim == 2 ): ylab = 'Z'

    fig = pyplot.figure()
    pyplot.clf()
    pyplot.hold(True)
    pyplot.plot(attractor[:,xdim],attractor[:,ydim],color='gray',linewidth=1)
    pyplot.xlabel(xlab,fontweight='bold',fontsize=12)
    pyplot.ylabel(ylab,fontweight='bold',fontsize=12)
    pyplot.title('Lorenz attractor',fontweight='bold',fontsize=14)
    if ( segment != None ):
        pyplot.plot(segment[:,0],segment[:,2],'ro',linewidth=2)
    pyplot.hold(False)
    return
# }}}

def plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=100):
# {{{
    '''
    Plot the Lorenz 1996 attractor in polar coordinates

    plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=100)

         obs - observations [None]
         ver - truth [None]
          xb - prior ensemble or ensemble mean [None]
          xa - posterior ensemble or ensemble mean [None]
           t - assimilation time [0]
           N - degrees of freedom to plot [1]
      figNum - figure handle [100]
    '''

    mean_dist = 35.0
    fig = pyplot.figure(figNum)
    pyplot.clf()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    theta = np.linspace(0.0,2*np.pi,N+1)
    pyplot.hold(True)

    if ( xb != None ):
        if ( len(xb.shape) == 1 ):
            tmp = np.zeros((N,1)) ; tmp[:,0] = xb ; xb = tmp
        for M in np.arange(0, xb.shape[1]):
            tmp = np.zeros(N+1) ; tmp[1:] = xb[:,M] ; tmp[0] = xb[-1,M]
            ax.plot(theta, tmp+mean_dist, 'b-')
    if ( xa != None ):
        if ( len(xa.shape) == 1 ):
            tmp = np.zeros((N,1)) ; tmp[:,0] = xa ; xa = tmp
        for M in np.arange(0, xa.shape[1]):
            tmp = np.zeros(N+1) ; tmp[1:] = xa[:,M] ; tmp[0] = xa[-1,M]
            ax.plot(theta, tmp+mean_dist, 'g-')
    if ( ver != None ):
        tmp = np.zeros(N+1) ; tmp[1:] = ver ; tmp[0]= ver[-1]
        ax.plot(theta, tmp+mean_dist, 'k-')
    if ( obs != None ):
        tmp = np.zeros(N+1) ; tmp[1:] = obs ; tmp[0] = obs[-1]
        ax.plot(theta, tmp+mean_dist, 'ro')

    ax.set_rmin(0.0)
    ax.set_rmax(mean_dist+25.0)
    rgrid  = np.arange(10,mean_dist+21,10)
    rgrid  = np.arange(10,mean_dist+20,10)
    rlabel = []
    rgrid, rlabel = pyplot.rgrids(rgrid, rlabel)

    tlabel = []
    tgrid  = np.arange(0,360,18)
    tlabel = np.arange(0,40,2)
    tgrid, tlabel = pyplot.thetagrids(tgrid, tlabel)

    pyplot.hold(False)
    title_str = 'k = %d' % (t)
    ax.set_title(title_str,fontweight='bold',fontsize=14)

    return
# }}}
