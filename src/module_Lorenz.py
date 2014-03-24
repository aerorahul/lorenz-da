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
import numpy
from   scipy       import integrate
from   matplotlib  import pyplot
from   netCDF4     import Dataset
###############################################################

module = 'module_Lorenz.py'

###############################################################
_private_vars = ['Name', 'Ndof', 'Par']
class Lorenz(object):
# {{{
    '''
    This module provides an interface to the Lorenz class models,
    together with its TL and Adjoint models.
    '''

    def __setattr__(self, key, val):
    # {{{
        '''
        prevent modification of read-only instance variables.
        '''
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to rebind read-only instance variable %s' % key)
        else:
            self.__dict__[key] = val
    #}}}

    def __delattr__(self, key):
    # {{{
        '''
        prevent deletion of read-only instance variables.
        '''
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError('Attempt to unbind read-only instance variable %s' % key)
        else:
            del self.__dict__[key]
    #}}}

    def __init__(self):
    # {{{
        '''
        Populates the Lorenz class given the model name,
        time-step, degrees of freedom and other model specific parameters
        '''
        pass
    #}}}

    def init(self,Name='L96',dt=1.0e-4,Ndof=None,Par=None):
    # {{{
        '''
        Populates the Lorenz class given the model name,
        time-step, degrees of freedom and other model specific parameters
        '''

        self.Name = Name
        self.dt   = dt
        if   ( self.Name == 'L63' ):
            self.Ndof = 3                if ( Ndof == None ) else Ndof
            self.Par = [10., 28., 8./3.] if ( Par  == None ) else Par
        elif ( self.Name == 'L96' ):
            self.Ndof = 40       if ( Ndof == None ) else Ndof
            self.Par = [8., 8.4] if ( Par  == None ) else Par
        else:
            raise AttributeError('Invalid model option %s' % self.Name)

    #}}}

    def advance(self, x0, t, perfect=True, result=None, **kwargs):
    # {{{
        '''
        advance - function that integrates the model state, given initial conditions 'x0'
        and length of the integration in t.
        The nature of the model advance as specified by 'perfect' and is only valid
        for L96 system.

        xs = advance(self, x0, t, perfect=True, **kwargs)

         self - model class for the model containing model static parameters
           x0 - initial state at time t = 0
            t - vector of time from t = [0, T]
      perfect - If perfect model run for L96, use self.Par[0], else use self.Par[1]
       result - result to be put back into, instead of normal return. To be used when multiprocessing
     **kwargs - any additional arguments that need to go in the model advance call
           xs - final state at time t = T
        '''

        if   ( self.Name == 'L63' ):
            par = self.Par
        elif ( self.Name == 'L96' ):
            if ( perfect ): par = self.Par[0]
            else:           par = self.Par[1]
        else:
            print '%s is an invalid model, exiting.' % self.Name
            sys.exit(1)

        exec('xs = integrate.odeint(self.%s, x0, t, (par, 0.0), **kwargs)' % (self.Name))

        if ( result == None ):
            return xs
        else:
            result.put(xs)
    # }}}

    def advance_tlm(self, x0, t, xref, tref, adjoint=False, perfect=True, result=None, **kwargs):
    # {{{
        '''
        advance_tlm - function that integrates the model state, using the TLM (or Adjoint)
        given initial conditions 'x0' and length of the integration in t.
        The nature of the model advance as specified by 'perfect' and is only valid
        for L96 system.

        xs = advance_tlm(self, x0, t, xref, tref, adjoint, perfect=True, **kwargs)

         self - model class for the model containing model static parameters
           x0 - initial state at time t = 0
            t - vector of time from t = [0, T]
         xref - non-linear reference trajectory
         tref - vector of time from t = [0, T]
      adjoint - adjoint (True) or forward TLM (False) [DEFAULT: False]
      perfect - If perfect model run for L96, use self.Par[0], else use self.Par[1]
       result - result to be put back into, instead of normal return. To be used when multiprocessing
     **kwargs - any additional arguments that need to go in the model advance call
           xs - final state at time t = T
        '''

        if   ( self.Name == 'L63' ):
            par = self.Par
        elif ( self.Name == 'L96' ):
            if ( perfect ): par = self.Par[0]
            else:           par = self.Par[1]
        else:
            print '%s is an invalid model, exiting.' % self.Name
            sys.exit(1)

        if ( adjoint ): xref = numpy.flipud(xref)

        exec('xs = integrate.odeint(self.%s_tlm, x0, t, (par,xref,tref,adjoint), **kwargs)' % self.Name)

        if ( result == None ):
            return xs
        else:
            result.put(xs)
    # }}}

    def L63(self, x0, t, par, dummy):
    # {{{
        '''
        L63 - function that integrates the Lorenz 1963 equations, given parameters 'par' and initial
        conditions 'x0'

        xs = L63(x0, t, (par, dummy))

         self - model class for the model containing model static parameters
           x0 - initial state at time t = 0
            t - vector of time from t = [0, T]
          par - parameters of the Lorenz 1963 system
        dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
           xs - final state at time t = T
        '''

        x, y, z = x0
        s, r, b = par

        x_dot = s*(y-x)
        y_dot = r*x -y -x*z
        z_dot = x*y - b*z

        xs = numpy.array([x_dot, y_dot, z_dot])

        return xs
    # }}}

    def L63_tlm(self, x0, t, par, xsave, tsave, adjoint):
    # {{{
        '''
        L63_tlm - function that integrates the Lorenz 1963 equations forward or backward using a TLM and
        its adjoint, given parameters 'par' and initial perturbations 'x0'

        xs = L63_tlm(x0, t, (par, xsave, tsave, adjoint))

       self - model class for the model containing model static parameters
         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
        par - parameters of the Lorenz 1963 system
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
         xs - evolved perturbations at time t = T
        '''

        s, r, b = par

        x = numpy.interp(t,tsave,xsave[:,0])
        y = numpy.interp(t,tsave,xsave[:,1])
        z = numpy.interp(t,tsave,xsave[:,2])

        M = numpy.array([[-s,   s,  0],
                         [r-z, -1, -x],
                         [y,    x, -b]])

        if ( adjoint ):
            xs = numpy.dot(numpy.transpose(M),x0)
        else:
            xs = numpy.dot(M,x0)

        return xs
    # }}}

    def L96(self, x0, t, F, dummy):
    # {{{
        '''
        L96 - function that integrates the Lorenz and Emanuel 1998 equations, given forcing 'F' and initial
        conditions 'x0'

        xs = L96(x0, t, (F, dummy))

     self - model class for the model containing model static parameters
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
        F - Forcing
    dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
       xs - final state at time t = T
        '''

        Ndof = len(x0)
        xs = numpy.zeros(Ndof)

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

    def L96_tlm(self, x0, t, F, xsave, tsave, adjoint):
    # {{{
        '''
        L96_tlm - function that integrates the Lorenz and Emanuel 1998 equations forward or backward
        using a TLM and its adjoint, given Forcing 'F' and initial perturbations 'x0'

        xs = L96_tlm(x0, t, (F, xsave, tsave, adjoint))

       self - model class for the model containing model static parameters
         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
          F - Forcing
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
         xs - evolved perturbations at time t = T
        '''

        Ndof = len(x0)
        x = numpy.zeros(Ndof)

        for j in range(0,Ndof):
            x[j] = numpy.interp(t,tsave,xsave[:,j])

        M = numpy.zeros((Ndof,Ndof))

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
            xs = numpy.dot(numpy.transpose(M),x0)
        else:
            xs = numpy.dot(M,x0)

        return xs
    # }}}

    def L96_IAU(x0, t, F, G):
    # {{{
        '''
        L96_IAU - function that integrates the Lorenz and Emanuel 1998 equations,
        given initial conditions 'x0', forcing 'F' and additional forcing 'G' (eg. IAU tendency)

        xs = L96(x0, t, (F, G))

           xs - final state at time t = T
           x0 - initial state at time t = 0
            t - vector of time from t = [0, T]
            F - Forcing
            G - additional forcing (eg. IAU tendency)
        '''

        Ndof = len(x0)
        xs = numpy.zeros(Ndof)

        for j in range(0,Ndof):
            jp1 = j + 1
            if ( jp1 >= Ndof ): jp1 = jp1 - Ndof
            jm2 = j - 2
            if ( jm2 < 0 ): jm2 = Ndof + jm2
            jm1 = j - 1
            if ( jm1 < 0 ): jm1 = Ndof + jm1

            xs[j] = ( x0[jp1] - x0[jm2] ) * x0[jm1] - x0[j] + F + G[j]

        return xs
    # }}}

# }}}
###############################################################

###############################################################
def plot_L63(obs=None, ver=None, xb=None, xa=None, xdim=0, ydim=2, **kwargs):
# {{{
    '''
    Plot the Lorenz 1963 attractor in 2D

    plot_L63(obs=None, ver=None, xb=None, xa=None, xdim=0, ydim=2, **kwargs)

        obs - x,y,z from t = [0, T]
        ver - x,y,z from t = [0, T]
         xb - prior x,y,z from t = [0, T]
         xa - posterior x,y,z from t = [0, T]
       xdim - variable along x-axis (X)
       ydim - variable along y-axis (Z)
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

    att = None
    pretitle = None
    for key in kwargs:
        if ( key == 'att' ): att = kwargs[key]
        if ( key == 'pretitle' ): pretitle = kwargs[key]

    if ( att != None ): pyplot.plot(att[:,xdim], att[:,ydim], color='gray', linewidth=1)
    if ( xb  != None ): pyplot.plot(xb[ :,xdim], xb[ :,ydim], 'b-', linewidth=1)
    if ( xa  != None ): pyplot.plot(xa[ :,xdim], xa[ :,ydim], 'r-', linewidth=1)
    if ( ver != None ): pyplot.plot(ver[:,xdim], ver[:,ydim], 'k-', linewidth=1)
    if ( obs != None ): pyplot.plot(obs[:,xdim], obs[:,ydim], 'yo', markeredgecolor='y')

    pyplot.xlabel(xlab,fontweight='bold',fontsize=12)
    pyplot.ylabel(ylab,fontweight='bold',fontsize=12)
    title_str = 'Lorenz attractor'
    pyplot.title(title_str,fontweight='bold',fontsize=14)
    fig.canvas.set_window_title(title_str)

    return fig
# }}}
###############################################################

###############################################################
def plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None, **kwargs):
# {{{
    '''
    Plot the Lorenz 1996 attractor in polar coordinates

    plot_L96(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None, **kwargs)

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
    pyplot.subplot(111, polar=True)
    theta = numpy.linspace(0.0,2*numpy.pi,N+1)
    pyplot.hold(True)

    # start by plotting a dummy tiny white dot dot at 0,0
    pyplot.plot(0, 0, 'w.', markeredgecolor='w', markersize=0.0)

    if ( xb != None ):
        if ( len(xb.shape) == 1 ):
            tmp = numpy.zeros(N+1) ; tmp[1:] = xb; tmp[0] = xb[-1]
        else:
            xmin, xmax, xmean = numpy.min(xb,axis=1), numpy.max(xb,axis=1), numpy.mean(xb,axis=1)
            tmpmin  = numpy.zeros(N+1) ; tmpmin[ 1:] = xmin;  tmpmin[ 0] = xmin[ -1]
            tmpmax  = numpy.zeros(N+1) ; tmpmax[ 1:] = xmax;  tmpmax[ 0] = xmax[ -1]
            tmp     = numpy.zeros(N+1) ; tmp[    1:] = xmean; tmp[0]     = xmean[-1]
            pyplot.fill_between(theta, tmpmin+mean_dist, tmpmax+mean_dist, facecolor='blue', edgecolor='blue', alpha=0.75)
        pyplot.plot(theta, tmp+mean_dist, 'b-', linewidth=2.0)
    if ( xa != None ):
        if ( len(xa.shape) == 1 ):
            tmp = numpy.zeros(N+1) ; tmp[1:] = xa; tmp[0] = xa[-1]
        else:
            xmin, xmax, xmean = numpy.min(xa,axis=1), numpy.max(xa,axis=1), numpy.mean(xa,axis=1)
            tmpmin  = numpy.zeros(N+1) ; tmpmin[ 1:] = xmin;  tmpmin[ 0] = xmin[ -1]
            tmpmax  = numpy.zeros(N+1) ; tmpmax[ 1:] = xmax;  tmpmax[ 0] = xmax[ -1]
            tmp     = numpy.zeros(N+1) ; tmp[    1:] = xmean; tmp[0]     = xmean[-1]
            pyplot.fill_between(theta, tmpmin+mean_dist, tmpmax+mean_dist, facecolor='red', edgecolor='red', alpha=0.5)
        pyplot.plot(theta, tmp+mean_dist, 'r-', linewidth=2.0)
    if ( ver != None ):
        tmp = numpy.zeros(N+1) ; tmp[1:] = ver ; tmp[0]= ver[-1]
        pyplot.plot(theta, tmp+mean_dist, 'k-', linewidth=2.0)
    if ( obs != None ):
        tmp = numpy.zeros(N+1) ; tmp[1:] = obs ; tmp[0] = obs[-1]
        pyplot.plot(theta, tmp+mean_dist, 'yo', markersize=7.5, markeredgecolor='y', alpha=0.95)

    pyplot.gca().set_rmin(0.0)
    pyplot.gca().set_rmax(mean_dist+25.0)
    rgrid  = numpy.array(numpy.linspace(10,mean_dist+25,5,endpoint=False),dtype=int)
    rlabel = []
    rgrid, rlabel = pyplot.rgrids(rgrid, rlabel)

    tlabel = []
    tgrid  = numpy.array(numpy.linspace(0,360,20,endpoint=False),dtype=int)
    tlabel = numpy.array(numpy.linspace(0, 40,20,endpoint=False),dtype=int)
    tgrid, tlabel = pyplot.thetagrids(tgrid, tlabel)

    pretitle = None
    for key in kwargs:
        if ( key == 'pretitle' ): pretitle = kwargs[key]

    if ( numpy.isreal(t) ): title = 'k = %d' % (t)
    else:                title = str(t)
    if ( not (pretitle == None) ): title = pretitle + ' - ' + title
    pyplot.title(title,fontweight='bold',fontsize=14)
    fig.canvas.set_window_title(title)

    return fig
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
            pert = 0.001 * ( numpy.random.randn(model.Ndof) )
            xa = xt + pert
        else:
            pert = 0.001 * ( numpy.random.randn(model.Ndof,Nens) )
            xa = numpy.transpose(xt + numpy.transpose(pert))
            xa = numpy.transpose(numpy.transpose(xa) - numpy.mean(xa,axis=1) + xt)

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
                xt = numpy.squeeze(nc.variables['truth'][read_index,])
                xa = numpy.transpose(numpy.squeeze(nc.variables['posterior'][read_index,]))
            nc.close()
        except Exception as Instance:
            print 'Exception occured during reading of %s' % (restart.filename)
            print type(Instance)
            print Instance.args
            print Instance
            sys.exit(1)

        if ( (len(numpy.shape(xa)) == 1) and (Nens != None) ):
            # populate initial ensemble analysis by perturbing the analysis and re-centering
            pert = 0.001 * ( numpy.random.randn(model.Ndof,Nens) )
            tmp = numpy.transpose(xa + numpy.transpose(pert))
            xa = numpy.transpose(numpy.transpose(tmp) - numpy.mean(tmp,axis=1) + xa)
        elif ( (len(numpy.shape(xa)) != 1) and (Nens != None) ):
            # populate initial ensemble analysis by picking a subset from the analysis ensemble
            if ( Nens <= numpy.shape(xa)[1] ):
                xa = numpy.squeeze(xa[:,0:Nens])
            else:
                print 'size(Xa) = [%d, %d]' % (numpy.shape(xa)[0], numpy.shape(xa)[1])
                sys.exit(1)
        elif ( (len(numpy.shape(xa)) != 1) and (Nens == None) ):
            xa = numpy.mean(xa, axis=1)

        return [xt,xa]

    # insure the same sequence of random numbers EVERY TIME
    numpy.random.seed(0)

    source = 'get_IC'

    print 'Generating ICs for %s' % model.Name

    if (   model.Name == 'L63' ):

        if ( restart.time == None ):
            print '... from Miller et al., 1994'

            xt = numpy.array([1.508870, -1.531271, 25.46091])

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    elif ( model.Name == 'L96' ):

        if ( restart.time == None ):
            print '... from Lorenz and Emanuel, 1998'

            xt    = numpy.ones(model.Ndof) * model.Par[0]
            xt[0] = 1.001 * model.Par[0]

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    return [xt, xa]
# }}}
###############################################################
