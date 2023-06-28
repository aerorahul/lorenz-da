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
from netCDF4 import Dataset
from matplotlib import pyplot
from scipy import integrate
import numpy
import sys
import os
__author__ = "Rahul Mahajan"
__email__ = "rahul.mahajan@nasa.gov"
__copyright__ = "Copyright 2011, NASA / GSFC / GMAO"
__license__ = "GPL"
__status__ = "Prototype"
###############################################################

###############################################################
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
            raise AttributeError(
                'Attempt to rebind read-only instance variable %s' %
                key)
        else:
            self.__dict__[key] = val
    # }}}

    def __delattr__(self, key):
        # {{{
        '''
        prevent deletion of read-only instance variables.
        '''
        if key in self.__dict__ and key in _private_vars:
            raise AttributeError(
                'Attempt to unbind read-only instance variable %s' %
                key)
        else:
            del self.__dict__[key]
    # }}}

    def __init__(self):
        # {{{
        '''
        Populates the Lorenz class given the model name,
        time-step, degrees of freedom and other model specific parameters
        '''
        pass
    # }}}

    def init(self, Name='L96', dt=1.0e-4, Ndof=None, Par=None):
        # {{{
        '''
        Populates the Lorenz class given the model name,
        time-step, degrees of freedom and other model specific parameters
        '''

        self.Name = Name
        self.dt = dt
        if (self.Name == 'L63'):
            self.Par = [10., 28., 8. / 3.] if (Par is None) else Par
            self.Ndof = 3 if (Ndof is None) else Ndof
        elif (self.Name == 'L96'):
            self.Par = [8., 8.4] if (Par is None) else Par
            self.Ndof = 40 if (Ndof is None) else Ndof
        elif (self.Name == 'L96_2scale'):
            self.Par = [
                8., 8.4, 40, 4, 10.0, 10.0, 1.0] if (
                Par is None) else Par
#                       F   F+dF m   n  c     b     h
            self.Ndof = self.Par[2] * \
                (self.Par[3] + 1) if (Ndof is None) else Ndof
        else:
            raise AttributeError('Invalid model option %s' % self.Name)

    # }}}

    def advance(self, x0, t, perfect=True, result=None, **kwargs):
        # {{{
        '''
        advance - function that integrates the model state, given initial conditions 'x0'
        and length of the integration in t.
        The nature of the model advance as specified by 'perfect' and is only valid
        for L96 system.

        xs = advance(self, x0, t, perfect=True, result=None, **kwargs)

         self - model class for the model containing model static parameters
           x0 - initial state at time t = 0
            t - vector of time from t = [0, T]
      perfect - If perfect model run for L96, use self.Par[0], else use self.Par[1]
       result - result to be put back into, instead of normal return. To be used when multiprocessing
     **kwargs - any additional arguments that need to go in the model advance call
           xs - final state at time t = T
        '''

        if (self.Name == 'L63'):
            func = self.L63
            par = None
        elif (self.Name == 'L96'):
            func = self.L96
            if (perfect):
                par = self.Par[0]
            else:
                par = self.Par[1]
        elif (self.Name == 'L96_2scale'):
            func = self.L96_2scale
            if (perfect):
                par = self.Par[0]
            else:
                par = self.Par[1]
        else:
            print(('%s is an invalid model, exiting.' % self.Name))
            sys.exit(1)

        # exec('xs = integrate.odeint(self.%s, x0, t, (par, 0.0), **kwargs)' % (self.Name))
        xs = integrate.odeint(func, x0, t, (par, 0.0), **kwargs)

        if (result is None):
            return xs
        else:
            result.put(xs)
    # }}}

    def advance_tlm(
            self,
            x0,
            t,
            xref,
            tref,
            adjoint=False,
            perfect=True,
            result=None,
            **kwargs):
        # {{{
        '''
        advance_tlm - function that integrates the model state, using the TLM (or Adjoint)
        given initial conditions 'x0' and length of the integration in t.
        The nature of the model advance as specified by 'perfect' and is only valid
        for L96 system.

        xs = advance_tlm(self, x0, t, xref, tref, adjoint, perfect=True, result=None, **kwargs)

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

        if (self.Name == 'L63'):
            func = self.L63_tlm
            par = None
        elif (self.Name == 'L96'):
            func = self.L96_tlm
            if (perfect):
                par = self.Par[0]
            else:
                par = self.Par[1]
        elif (self.Name == 'L96_2scale'):
            func = self.L96_2scale_tlm
            if (perfect):
                par = self.Par[0]
            else:
                par = self.Par[1]
        else:
            print(('%s is an invalid model, exiting.' % self.Name))
            sys.exit(1)

        if (adjoint):
            xref = numpy.flipud(xref)

        xs = integrate.odeint(
            func, x0, t, (par, xref, tref, adjoint), **kwargs)

        if (result is None):
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
        s, r, b = self.Par

        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z

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

        s, r, b = self.Par

        x = numpy.interp(t, tsave, xsave[:, 0])
        y = numpy.interp(t, tsave, xsave[:, 1])
        z = numpy.interp(t, tsave, xsave[:, 2])

        M = numpy.array([[-s, s, 0],
                         [r - z, -1, -x],
                         [y, x, -b]])

        if (adjoint):
            xs = numpy.dot(numpy.transpose(M), x0)
        else:
            xs = numpy.dot(M, x0)

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

        xs = numpy.zeros(self.Ndof)

        for j in range(self.Ndof):
            jp1 = j + 1
            if (jp1 >= self.Ndof):
                jp1 = jp1 - self.Ndof
            jm2 = j - 2
            if (jm2 < 0):
                jm2 = jm2 + self.Ndof
            jm1 = j - 1
            if (jm1 < 0):
                jm1 = jm1 + self.Ndof

            xs[j] = (x0[jp1] - x0[jm2]) * x0[jm1] - x0[j] + F

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

        x = numpy.zeros(self.Ndof)

        for j in range(self.Ndof):
            x[j] = numpy.interp(t, tsave, xsave[:, j])

        M = numpy.zeros((self.Ndof, self.Ndof))

        for j in range(self.Ndof):
            jp1 = j + 1
            if (jp1 >= self.Ndof):
                jp1 = jp1 - self.Ndof
            jm2 = j - 2
            if (jm2 < 0):
                jm2 = jm2 + self.Ndof
            jm1 = j - 1
            if (jm1 < 0):
                jm1 = jm1 + self.Ndof

            M[j, jm2] = -x[jm1]
            M[j, jm1] = x[jp1] - x[jm2]
            M[j, j] = -1
            M[j, jp1] = x[jm1]

        if (adjoint):
            xs = numpy.dot(numpy.transpose(M), x0)
        else:
            xs = numpy.dot(M, x0)

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

        xs = numpy.zeros(self.Ndof)

        for j in range(self.Ndof):
            jp1 = j + 1
            if (jp1 >= self.Ndof):
                jp1 = jp1 - self.Ndof
            jm2 = j - 2
            if (jm2 < 0):
                jm2 = jm2 + self.Ndof
            jm1 = j - 1
            if (jm1 < 0):
                jm1 = jm1 + self.Ndof

            xs[j] = (x0[jp1] - x0[jm2]) * x0[jm1] - x0[j] + F + G[j]

        return xs
    # }}}

    def L96_2scale(self, x0, t, F, dummy):
        # {{{
        '''
        L96_2scale - function that integrates the Lorenz 1996 2 scale equations,
                     given parameters 'par' and initial conditions 'x0'

        xs = L96_2scale(x0, t, (par, dummy))

     self - model class for the model containing model static parameters
       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
        F - Forcing
    dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
       xs - final state at time t = T
        '''

        m, n = self.Par[:2]
        b, c, h = self.Par[4:]
        xs = numpy.zeros(self.Ndof)

        # first small scale
        js = m
        je = m * (n + 1)
        for j in range(js, je):
            jp1 = j + 1
            if (jp1 >= je):
                jp1 = jp1 - je + js
            jp2 = j + 2
            if (jp2 >= je):
                jp2 = jp2 - je + js
            jm1 = j - 1
            if (jm1 < js):
                jm1 = jm1 + je - js

            k = (j - js) / n

            xs[j] = (c * b) * x0[jp1] * (x0[jm1] - x0[jp2]) - \
                c * x0[j] + (h * c / b) * x0[k]

        # second large scale
        ks = 0
        ke = m
        for k in range(ks, ke):
            kp1 = k + 1
            if (kp1 >= ke):
                kp1 = kp1 - ke + ks
            km2 = k - 2
            if (km2 < ks):
                km2 = km2 + ke + ks
            km1 = k - 1
            if (km1 < ks):
                km1 = km1 + ke + ks

            js = m + n * k
            je = m + n * (k + 1)
            fast_sum = numpy.sum(x0[js:je])

            xs[k] = (x0[kp1] - x0[km2]) * x0[km1] - \
                x0[k] + F - (h * c / b) * fast_sum

        return xs
    # }}}

    def L96_2scale_tlm(self, x0, t, F, xsave, tsave, adjoint):
        # {{{
        '''
        L96_2scale_tlm - function that integrates the Lorenz 92 2-scale system forward or backward
        using a TLM and its adjoint, given parameters 'par' and initial perturbations 'x0'

        xs = L96_2scale_tlm(x0, t, (par, xsave, tsave, adjoint))

       self - model class for the model containing model static parameters
         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
          F - Forcing
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
         xs - evolved perturbations at time t = T
        '''

        m, n = self.Par[:2]
        b, c, h = self.Par[4:]
        x = numpy.zeros(self.Ndof)

        for j in range(self.Ndof):
            x[j] = numpy.interp(t, tsave, xsave[:, j])

        M = numpy.zeros((self.Ndof, self.Ndof))

        # first large scale
        ks = 0
        ke = m
        for k in range(ks, ke):
            kp1 = k + 1
            if (kp1 >= ke):
                kp1 = kp1 - ke + ks
            km2 = k - 2
            if (km2 < ks):
                km2 = km2 + ke + ks
            km1 = k - 1
            if (km1 < ks):
                km1 = km1 + ke + ks

            js = m + n * k
            je = m + n * (k + 1)

            M[k, km2] = -x[km1]
            M[k, km1] = x[kp1] - x[km2]
            M[k, k] = -1
            M[k, kp1] = x[km1]
            M[k, js:je] = -(h * c / b)

        # second small scale
        js = m
        je = m * (n + 1)
        for j in range(js, je):
            jp1 = j + 1
            if (jp1 >= je):
                jp1 = jp1 - je + js
            jp2 = j + 2
            if (jp2 >= je):
                jp2 = jp2 - je + js
            jm1 = j - 1
            if (jm1 < js):
                jm1 = jm1 + je - js

            ks = 0
            ke = m

            M[j, jm1] = (c * b) * x[jp1]
            M[j, j] = -c
            M[j, jp1] = (c * b) * (x[jm1] - x[jp2])
            M[j, jp2] = -(c * b) * x[jp1]
            M[j, ks:ke] = h * c / b

        if (adjoint):
            xs = numpy.dot(numpy.transpose(M), x0)
        else:
            xs = numpy.dot(M, x0)

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

    if (xdim == ydim):
        xdim = 0
        ydim = 2

    if (xdim < 0 or xdim > 2):
        xdim = 0
    if (ydim < 0 or ydim > 2):
        ydim = 2

    if (xdim == 0):
        xlab = 'X'
    elif (xdim == 1):
        xlab = 'Y'
    elif (xdim == 2):
        xlab = 'Z'

    if (ydim == 0):
        ylab = 'X'
    elif (ydim == 1):
        ylab = 'Y'
    elif (ydim == 2):
        ylab = 'Z'

    fig = pyplot.figure()
    pyplot.clf()

    att = None
    pretitle = None
    for key in kwargs:
        if (key == 'att'):
            att = kwargs[key]
        if (key == 'pretitle'):
            pretitle = kwargs[key]

    if (att is not None):
        pyplot.plot(att[:, xdim], att[:, ydim], color='gray', linewidth=1)
    if (xb is not None):
        pyplot.plot(xb[:, xdim], xb[:, ydim], 'b-', linewidth=1)
    if (xa is not None):
        pyplot.plot(xa[:, xdim], xa[:, ydim], 'r-', linewidth=1)
    if (ver is not None):
        pyplot.plot(ver[:, xdim], ver[:, ydim], 'k-', linewidth=1)
    if (obs is not None):
        pyplot.plot(obs[:, xdim], obs[:, ydim], 'yo', markeredgecolor='y')

    pyplot.xlabel(xlab, fontweight='bold', fontsize=12)
    pyplot.ylabel(ylab, fontweight='bold', fontsize=12)
    title_str = 'Lorenz attractor'
    pyplot.title(title_str, fontweight='bold', fontsize=14)

    return fig
# }}}
###############################################################

###############################################################


def plot_L96(
        obs=None,
        ver=None,
        xb=None,
        xa=None,
        t=0,
        N=1,
        figNum=None,
        **kwargs):
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

    if (figNum is None):
        fig = pyplot.figure()
    else:
        fig = pyplot.figure(figNum)
    pyplot.clf()
    mean_dist = 35.0
    pyplot.subplot(111, polar=True)
    theta = numpy.linspace(0.0, 2 * numpy.pi, N + 1)

    # start by plotting a dummy tiny white dot dot at 0,0
    pyplot.plot(0, 0, 'w.', markeredgecolor='w', markersize=0.0)

    if (xb is not None):
        if (len(xb.shape) == 1):
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xb
            tmp[0] = xb[-1]
        else:
            xmin, xmax, xmean = numpy.min(
                xb, axis=1), numpy.max(
                xb, axis=1), numpy.mean(
                xb, axis=1)
            tmpmin = numpy.zeros(N + 1)
            tmpmin[1:] = xmin
            tmpmin[0] = xmin[-1]
            tmpmax = numpy.zeros(N + 1)
            tmpmax[1:] = xmax
            tmpmax[0] = xmax[-1]
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xmean
            tmp[0] = xmean[-1]
            pyplot.fill_between(
                theta,
                tmpmin + mean_dist,
                tmpmax + mean_dist,
                facecolor='blue',
                edgecolor='blue',
                alpha=0.75)
        pyplot.plot(theta, tmp + mean_dist, 'b-', linewidth=2.0)
    if (xa is not None):
        if (len(xa.shape) == 1):
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xa
            tmp[0] = xa[-1]
        else:
            xmin, xmax, xmean = numpy.min(
                xa, axis=1), numpy.max(
                xa, axis=1), numpy.mean(
                xa, axis=1)
            tmpmin = numpy.zeros(N + 1)
            tmpmin[1:] = xmin
            tmpmin[0] = xmin[-1]
            tmpmax = numpy.zeros(N + 1)
            tmpmax[1:] = xmax
            tmpmax[0] = xmax[-1]
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xmean
            tmp[0] = xmean[-1]
            pyplot.fill_between(
                theta,
                tmpmin + mean_dist,
                tmpmax + mean_dist,
                facecolor='red',
                edgecolor='red',
                alpha=0.5)
        pyplot.plot(theta, tmp + mean_dist, 'r-', linewidth=2.0)
    if (ver is not None):
        tmp = numpy.zeros(N + 1)
        tmp[1:] = ver
        tmp[0] = ver[-1]
        pyplot.plot(theta, tmp + mean_dist, 'k-', linewidth=2.0)
    if (obs is not None):
        tmp = numpy.zeros(N + 1)
        tmp[1:] = obs
        tmp[0] = obs[-1]
        pyplot.plot(
            theta,
            tmp + mean_dist,
            'yo',
            markersize=7.5,
            markeredgecolor='y',
            alpha=0.95)

    pyplot.gca().set_rmin(0.0)
    pyplot.gca().set_rmax(mean_dist + 25.0)
    rgrid = numpy.array(
        numpy.linspace(
            10,
            mean_dist + 25,
            5,
            endpoint=False),
        dtype=int)
    rlabel = []
    rgrid, rlabel = pyplot.rgrids(rgrid, rlabel)

    tlabel = []
    tgrid = numpy.array(numpy.linspace(0, 360, 20, endpoint=False), dtype=int)
    tlabel = numpy.array(numpy.linspace(0, 40, 20, endpoint=False), dtype=int)
    tgrid, tlabel = pyplot.thetagrids(tgrid, tlabel)

    pretitle = None
    for key in kwargs:
        if (key == 'pretitle'):
            pretitle = kwargs[key]

    if (numpy.isreal(t)):
        title = 'k = %d' % (t)
    else:
        title = str(t)
    if (not (pretitle is None)):
        title = pretitle + ' - ' + title
    pyplot.title(title, fontweight='bold', fontsize=14)

    return fig
# }}}
###############################################################

###############################################################


def plot_L96_2scale(
        obs=None,
        ver=None,
        xb=None,
        xa=None,
        t=0,
        N=1,
        figNum=None,
        **kwargs):
    # {{{
    '''
    Plot the Lorenz 1996 2 scale attractor in polar coordinates

    plot_L96_2scale(obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None, **kwargs)

         obs - observations [None]
         ver - truth [None]
          xb - prior ensemble or ensemble mean [None]
          xa - posterior ensemble or ensemble mean [None]
           t - assimilation time [0]
           N - degrees of freedom to plot [1]
      figNum - figure handle [None]
    '''

    if (figNum is None):
        fig = pyplot.figure()
    else:
        fig = pyplot.figure(figNum)
    pyplot.clf()
    mean_dist_x = 35.0
    mean_dist_y = 75.0
    pyplot.subplot(111, polar=True)
    theta = numpy.linspace(0.0, 2 * numpy.pi, N + 1)

    # start by plotting a dummy tiny white dot dot at 0,0
    pyplot.plot(0, 0, 'w.', markeredgecolor='w', markersize=0.0)

    if (xb is not None):
        if (len(xb.shape) == 1):
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xb
            tmp[0] = xb[-1]
        else:
            xmin, xmax, xmean = numpy.min(
                xb, axis=1), numpy.max(
                xb, axis=1), numpy.mean(
                xb, axis=1)
            tmpmin = numpy.zeros(N + 1)
            tmpmin[1:] = xmin
            tmpmin[0] = xmin[-1]
            tmpmax = numpy.zeros(N + 1)
            tmpmax[1:] = xmax
            tmpmax[0] = xmax[-1]
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xmean
            tmp[0] = xmean[-1]
            pyplot.fill_between(
                theta,
                tmpmin + mean_dist,
                tmpmax + mean_dist,
                facecolor='blue',
                edgecolor='blue',
                alpha=0.75)
        pyplot.plot(theta, tmp + mean_dist, 'b-', linewidth=2.0)
    if (xa is not None):
        if (len(xa.shape) == 1):
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xa
            tmp[0] = xa[-1]
        else:
            xmin, xmax, xmean = numpy.min(
                xa, axis=1), numpy.max(
                xa, axis=1), numpy.mean(
                xa, axis=1)
            tmpmin = numpy.zeros(N + 1)
            tmpmin[1:] = xmin
            tmpmin[0] = xmin[-1]
            tmpmax = numpy.zeros(N + 1)
            tmpmax[1:] = xmax
            tmpmax[0] = xmax[-1]
            tmp = numpy.zeros(N + 1)
            tmp[1:] = xmean
            tmp[0] = xmean[-1]
            pyplot.fill_between(
                theta,
                tmpmin + mean_dist,
                tmpmax + mean_dist,
                facecolor='red',
                edgecolor='red',
                alpha=0.5)
        pyplot.plot(theta, tmp + mean_dist, 'r-', linewidth=2.0)
    if (ver is not None):
        tmp = numpy.zeros(N + 1)
        tmp[1:] = ver
        tmp[0] = ver[-1]
        pyplot.plot(theta, tmp + mean_dist, 'k-', linewidth=2.0)
    if (obs is not None):
        tmp = numpy.zeros(N + 1)
        tmp[1:] = obs
        tmp[0] = obs[-1]
        pyplot.plot(
            theta,
            tmp + mean_dist,
            'yo',
            markersize=7.5,
            markeredgecolor='y',
            alpha=0.95)

    pyplot.gca().set_rmin(0.0)
    pyplot.gca().set_rmax(mean_dist_y + 25.0)
    rgrid = numpy.array(
        numpy.linspace(
            10,
            mean_dist + 25,
            5,
            endpoint=False),
        dtype=int)
    rlabel = []
    rgrid, rlabel = pyplot.rgrids(rgrid, rlabel)

    tlabel = []
    tgrid = numpy.array(numpy.linspace(0, 360, 20, endpoint=False), dtype=int)
    tlabel = numpy.array(numpy.linspace(0, 40, 20, endpoint=False), dtype=int)
    tgrid, tlabel = pyplot.thetagrids(tgrid, tlabel)

    pretitle = None
    for key in kwargs:
        if (key == 'pretitle'):
            pretitle = kwargs[key]

    if (numpy.isreal(t)):
        title = 'k = %d' % (t)
    else:
        title = str(t)
    if (not (pretitle is None)):
        title = pretitle + ' - ' + title
    pyplot.title(title, fontweight='bold', fontsize=14)

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

    def perturb_truth(xt, model, Nens=None, pert_scale=1.e-3):
        '''
        populate initial ensemble analysis by perturbing true state and recentering
        '''

        if Nens is None:
            xa = xt + pert_scale * numpy.random.randn(model.Ndof)
        else:
            perts = pert_scale * numpy.random.randn(Nens, model.Ndof)
            xa = xt + perts
            xa = (xa - numpy.mean(xa, axis=0) + xt).T

        return xa

    def read_from_restart(restart, Nens=None):
        '''
        read from a specified restart file
        '''

        if not os.path.isfile(restart.filename):
            print(('ERROR : %s does not exist ' % restart.filename))
            sys.exit(2)

        try:
            nc = Dataset(restart.filename, mode='r', format='NETCDF4')
            ntime = len(nc.dimensions['ntime'])
            if (restart.time == 0):
                read_index = 0
            elif (restart.time > 0):
                read_index = restart.time - 1
            elif (restart.time < 0):
                read_index = ntime + restart.time
            if ((read_index < 0) or (read_index >= ntime)):
                print(('ERROR : t = %d does not exist in %s' %
                      (read_index + 1, restart.filename)))
                print(('        valid options are t = +/- [1 ... %d]' % ntime))
                sys.exit(2)
            else:
                print(('... from t = %d in %s' %
                       (read_index + 1, restart.filename)))
                xt = numpy.squeeze(nc.variables['truth'][read_index,])
                xa = numpy.transpose(numpy.squeeze(
                    nc.variables['posterior'][read_index,]))
            nc.close()
        except Exception as Instance:
            print(('Exception occured during reading of %s' % (restart.filename)))
            print((type(Instance)))
            print((Instance.args))
            print(Instance)
            sys.exit(1)

        if ((len(numpy.shape(xa)) == 1) and (Nens is not None)):
            # populate initial ensemble analysis by perturbing the analysis and
            # re-centering
            pert = 0.001 * (numpy.random.randn(model.Ndof, Nens))
            tmp = numpy.transpose(xa + numpy.transpose(pert))
            xa = numpy.transpose(
                numpy.transpose(tmp) -
                numpy.mean(
                    tmp,
                    axis=1) +
                xa)
        elif ((len(numpy.shape(xa)) != 1) and (Nens is not None)):
            # populate initial ensemble analysis by picking a subset from the
            # analysis ensemble
            if (Nens <= numpy.shape(xa)[1]):
                xa = numpy.squeeze(xa[:, 0:Nens])
            else:
                print(('size(Xa) = [%d, %d]' %
                       (numpy.shape(xa)[0], numpy.shape(xa)[1])))
                sys.exit(1)
        elif ((len(numpy.shape(xa)) != 1) and (Nens is None)):
            xa = numpy.mean(xa, axis=1)

        return [xt, xa]

    # insure the same sequence of random numbers EVERY TIME
    numpy.random.seed(0)

    source = 'get_IC'

    print(('Generating ICs for %s' % model.Name))

    if (model.Name == 'L63'):

        if (restart.time is None):
            print('... from Miller et al., 1994')

            xt = numpy.array([1.508870, -1.531271, 25.46091])

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    elif (model.Name == 'L96'):

        if (restart.time is None):
            print('... from Lorenz and Emanuel, 1998')

            xt = numpy.ones(model.Ndof) * model.Par[0]
            xt[0] = 1.001 * model.Par[0]

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    elif (model.Name == 'L96_2scale'):

        if (restart.time is None):
            print('... from Lorenz 1996 2 scale')

            xt = numpy.zeros(model.Ndof)
            xt[:model.Par[0]] = numpy.ones(model.Par[0]) * model.Par[2]
            xt[0] = 1.001 * model.Par[2]

            xt[model.Par[0]:] = 0.01 * xt[1]
            xt[model.Par[0]::model.Par[1]] = 0.011 * xt[1]

            xa = perturb_truth(xt, model, Nens=Nens)

        else:

            [xt, xa] = read_from_restart(restart, Nens=Nens)

    return [xt, xa]
# }}}
###############################################################
