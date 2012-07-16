#!/usr/bin/env python

###############################################################
# < next few lines under version control, D O  N O T  E D I T >
# $Date$
# $Revision$
# $Author$
# $Id$
###############################################################

###############################################################
# module_Lorenz.py - Functions related to the Lorenz models
#                    L63 and L96
###############################################################

###############################################################
__author__    = "Rahul Mahajan"
__email__     = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__   = "GPL"
__status__    = "Prototype"
###############################################################

###############################################################
import os
import sys
import numpy       as     np
from   scipy       import integrate
from   matplotlib  import pyplot
from   netCDF4     import Dataset
from module_Lorenz import *
###############################################################

module = 'module_Lorenz.py'

###############################################################
def advance_model(model, x0, t, perfect=True, **kwargs):
# {{{
    '''
    advance_model - function that integrates the model state, given initial conditions 'x0'
    and length of the integration in t.
    The nature of the model advance as specified by 'perfect' and is only valid
    for L96 system.

    xs = advance_model(model, x0, t, perfect=True)

       xs - final state at time t = T
    model - model class for the model containing model static parameters
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
  perfect - If perfect model run for L96, use model.Par[0], else use model.Par[1]
    '''

    if   ( model.Name == 'L63' ):
        par = model.Par
    elif ( model.Name == 'L96' ):
        if ( perfect ): par = model.Par[0]
        else:           par = model.Par[1]
    else:
        print '%s is an invalid model, exiting.' % model.Name
        sys.exit(1)

    exec('xs = integrate.odeint(%s, x0, t, (par, 0.0), **kwargs)' % (model.Name))

    return xs
# }}}
###############################################################

###############################################################
def advance_model_tlm(model, x0, t, xref, tref, adjoint=False, perfect=True, **kwargs):
# {{{
    '''
    advance_model_tlm - function that integrates the model state, using the TLM (or Adjoint)
    given initial conditions 'x0' and length of the integration in t.
    The nature of the model advance as specified by 'perfect' and is only valid
    for L96 system.

    xs = advance_model_tlm(model, x0, t, xref, tref, adjoint, perfect=True)

       xs - final state at time t = T
    model - model class for the model containing model static parameters
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
     xref - non-linear reference trajectory
     tref - vector of time from t = [0, T]
  adjoint - adjoint (True) or forward TLM (False) [DEFAULT: False]
  perfect - If perfect model run for L96, use model.Par[0], else use model.Par[1]
    '''

    if   ( model.Name == 'L63' ):
        par = model.Par
    elif ( model.Name == 'L96' ):
        if ( perfect ): par = model.Par[0]
        else:           par = model.Par[1]
    else:
        print '%s is an invalid model, exiting.' % model.Name
        sys.exit(1)

    if ( adjoint ): xref = np.flipud(xref)

    exec('xs = integrate.odeint(%s_tlm, x0, t, (par,xref,tref,adjoint), **kwargs)' % model.Name)

    return xs
# }}}
###############################################################

###############################################################
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
###############################################################

###############################################################
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
###############################################################

###############################################################
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
###############################################################

###############################################################
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
###############################################################

###############################################################
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
###############################################################

###############################################################
def plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None):
# {{{
    '''
    Plot the Lorenz 1996 attractor in polar coordinates

    plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None)

         obs - observations [None]
         ver - truth [None]
          xb - prior ensemble or ensemble mean [None]
          xa - posterior ensemble or ensemble mean [None]
           t - assimilation time [0]
           N - degrees of freedom to plot [1]
      figNum - figure handle [None]
    '''

    if ( figNum == None ):
        fig = pyplot.figure()
    else:
        fig = pyplot.figure(figNum)
    pyplot.clf()
    mean_dist = 35.0
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    theta = np.linspace(0.0,2*np.pi,N+1)
    pyplot.hold(True)

    if ( xb != None ):
        if ( len(xb.shape) == 1 ):
            tmp = np.zeros((N,1)) ; tmp[:,0] = xb ; xb = tmp
        for M in range(0, xb.shape[1]):
            tmp = np.zeros(N+1) ; tmp[1:] = xb[:,M] ; tmp[0] = xb[-1,M]
            ax.plot(theta, tmp+mean_dist, 'b-')
    if ( xa != None ):
        if ( len(xa.shape) == 1 ):
            tmp = np.zeros((N,1)) ; tmp[:,0] = xa ; xa = tmp
        for M in range(0, xa.shape[1]):
            tmp = np.zeros(N+1) ; tmp[1:] = xa[:,M] ; tmp[0] = xa[-1,M]
            ax.plot(theta, tmp+mean_dist, 'r-')
    if ( ver != None ):
        tmp = np.zeros(N+1) ; tmp[1:] = ver ; tmp[0]= ver[-1]
        ax.plot(theta, tmp+mean_dist, 'k-', linewidth=2)
    if ( obs != None ):
        tmp = np.zeros(N+1) ; tmp[1:] = obs ; tmp[0] = obs[-1]
        ax.plot(theta, tmp+mean_dist, 'yo', markeredgecolor='y')

    ax.set_rmin(0)
    ax.set_rmax(mean_dist+25)
    rgrid  = np.array(np.linspace(10,mean_dist+25,5,endpoint=False),dtype=int)
    rlabel = []
    rgrid, rlabel = pyplot.rgrids(rgrid, rlabel)

    tlabel = []
    tgrid  = np.array(np.linspace(0,360,20,endpoint=False),dtype=int)
    tlabel = np.array(np.linspace(0, 40,20,endpoint=False),dtype=int)
    tgrid, tlabel = pyplot.thetagrids(tgrid, tlabel)

    pyplot.hold(False)
    title_str = 'k = %d' % (t)
    ax.set_title(title_str,fontweight='bold',fontsize=14)

    return
# }}}
###############################################################

###############################################################
def get_IC(model, restart, Nens=None):
# {{{
    '''
    Get initial conditions based on model and restart conditions

    [xt, xa] = get_IC(model, restart, Nens=None)

        model - model Class
      restart - restart Class
         Nens - no. of ensemble members [None]
           xt - truth
           xa - analysis or analysis ensemble
    '''

    def perturb_truth(xt, model, Nens=None):
        '''
        populate initial ensemble analysis by perturbing true state and recentering
        '''

        if ( Nens == None ):
            pert = 0.001 * ( np.random.randn(model.Ndof) )
            xa = xt + pert
        else:
            pert = 0.001 * ( np.random.randn(model.Ndof,Nens) )
            xa = np.transpose(xt + np.transpose(pert))
            xa = np.transpose(np.transpose(xa) - np.mean(xa,axis=1) + xt)

        return xa

    def read_from_restart(restart, Nens=None):
        '''
        read from a specified restart file
        '''

        if not os.path.isfile(restart.filename):
            print 'ERROR : %s does not exist ' % restart.filename
            sys.exit(2)

        try:
            nc = Dataset(restart.filename, mode='r', format='NETCDF4')
            ntime = len(nc.dimensions['ntime'])
            if   ( restart.time == 0 ): read_index = 0
            elif ( restart.time >  0 ): read_index = restart.time - 1
            elif ( restart.time <  0 ): read_index = ntime + restart.time
            if ( (read_index < 0) or (read_index >= ntime) ):
                print 'ERROR : t = %d does not exist in %s' % (read_index+1, restart.filename)
                print '        valid options are t = +/- [1 ... %d]' % ntime
                sys.exit(2)
            else:
                print '... from t = %d in %s' % (read_index+1, restart.filename)
                xt = np.squeeze(nc.variables['truth'][read_index,])
                xa = np.transpose(np.squeeze(nc.variables['posterior'][read_index,]))
            nc.close()
        except Exception as Instance:
            print 'Exception occured during reading of %s' % (restart.filename)
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

        if ( (len(np.shape(xa)) == 1) and (Nens != None) ):
            # populate initial ensemble analysis by perturbing the analysis and re-centering
            pert = 0.001 * ( np.random.randn(model.Ndof,Nens) )
            tmp = np.transpose(xa + np.transpose(pert))
            xa = np.transpose(np.transpose(tmp) - np.mean(tmp,axis=1) + xa)
        elif ( (len(np.shape(xa)) != 1) and (Nens != np.shape(xa)[1]) ):
            # populate initial ensemble analysis by picking a subset from the analysis ensemble
            if ( Nens <= np.shape(xa)[1] ):
                xa = np.squeeze(xa[:,0:Nens])
            else:
                print 'size(Xa) = [%d, %d]' % (np.shape(xa)[0], np.shape(xa)[1])
                sys.exit(1)

        return [xt,xa]

    # insure the same sequence of random numbers EVERY TIME
    np.random.seed(0)

    source = 'get_IC'

    print 'Generating ICs for %s' % model.Name

    if (   model.Name == 'L63' ):

        if ( restart.time == None ):
            print '... from Miller et al., 1994'

            xt = np.array([1.508870, -1.531271, 25.46091])

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    elif ( model.Name == 'L96' ):

        if ( restart.time == None ):
            print '... from Lorenz and Emanuel, 1998'

            xt    = np.ones(model.Ndof) * model.Par[0]
            xt[0] = 1.001 * model.Par[0]

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    return [xt, xa]
# }}}
###############################################################
