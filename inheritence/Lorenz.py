import os
import numpy as np
from scipy import integrate
from netCDF4 import Dataset
import matplotlib.pyplot as plt

__all__ = ['LorenzBase', 'L63', 'L96']

from .Model import ModelBase


class LorenzBase(ModelBase):
    """
    This module provides an interface to the base Lorenz class models,
    together with its TL and Adjoint models.
    """

    def __init__(self, modelConfig):
        """
        Populates the Lorenz class given the model name,
        time-step, degrees of freedom and other model specific parameters
        """
        super().__init__(modelConfig)
        self.Ndof = modelConfig.get('Ndof')
        self.Par = modelConfig.get('Par')

    def advance(self, modelFunc, x0, t, *args, **kwargs):
        """
        method that integrates the model state x0(t0) -> xt(T) for Lorenz models,
        using the appropriate method provided by the sub-class 'modelFunc',
        initial conditions 'x0' and length of the integration in 't'.
        Additional arguments are provided via kwargs.
        Lorenz models use scipy.integrate.odeint to advance state

        xt = advance(self, modelFunc, x0, t, *args, **kwargs)

           :param modelFunc: RHS of the model
           :param x0: initial state at time t = t0
           :param t: vector of time from t = [t0, T]
           :param args: additional arguments to modelFunc
           :param kwargs: additional keyword arguments to odeint
           :return: final state at time t = T
        """

        xt = integrate.odeint(modelFunc, x0, t, *args, **kwargs)
        return xt

    def perturb_state(self, xt, amplitude=1.e-3, Nens=1):
        """
        perturb a state given the perturbation amplitude
        """

        perts = amplitude * np.random.randn(Nens, self.Ndof)
        xa = xt + perts
        xa = (xa - np.mean(xa, axis=0) + xt).T
        return np.squeeze(xa)

    def read_from_restart(self, restart, Nens=1):
        """
        read from a specified restart file
        """

        filename = restart.get('filename', 'default.nc4')
        time = restart.get('time', 0)

        if not os.path.isfile(filename):
            msg = 'ERROR : %s does not exist ' % filename
            raise IOError(msg)

        nc = Dataset(filename, mode='r', format='NETCDF4')
        ntime = len(nc.dimensions['ntime'])
        if time == 0:
            indx = 0
        elif time > 0:
            indx = time - 1
        elif time < 0:
            indx = ntime + time
        if (indx < 0) or (indx >= ntime):
            msg = 'ERROR : t = %d does not exist in %s\n' % (
                indx + 1, filename)
            msg += '        valid options are t = +/- [1 ... %d]' % ntime
            raise IndexError(msg)
        else:
            print(('... from t = %d in %s' % (indx + 1, filename)))
            xt = np.squeeze(nc.variables['truth'][indx, ])
            xa = np.transpose(np.squeeze(nc.variables['posterior'][indx, ]))
        nc.close()

        if len(np.shape(xa)) == 1:
            # populate initial ensemble by perturbing the analysis and
            # re-centering
            xa = self.perturb_state(xa, amplitude=0.001, Nens=Nens)

        elif len(np.shape(xa)) != 1:
            if Nens is None:  # return the mean of the ensemble
                xa = np.mean(xa, axis=1)
            elif Nens <= np.shape(xa)[1]:  # subset the ensemble
                xa = np.squeeze(xa[:, :Nens])
            else:
                msg = 'size(Xa) = [%d, %d]' % (
                    np.shape(xa)[0], np.shape(xa)[1])
                raise ValueError(msg)

        return [xt, xa]


class L63(LorenzBase):
    """
    This module provides an interface to Lorenz 63 model
    together with its TL and Adjoint models.
    """

    def __init__(self, modelConfig):
        """
        Populates the L63 class given the model configuration containing,
        name, time-step, degrees of freedom and other model specific parameters
        """
        super().__init__(modelConfig)

    def rhs(self, x0, t):
        """
        method that describes the RHS of Lorenz 1963 equations,
        given initial conditions 'x0'

        xt = L63.rhs(x0, t)

        x0 - initial state at time t = t0
         t - vector of time from t = [t0, T]
        xt - final state at time t = T
        """

        x, y, z = x0
        s, r, b = self.Par

        x_dot = s * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z

        xt = np.array([x_dot, y_dot, z_dot])

        return xt

    def rhs_tlm(self, x0, t, xsave, tsave, adjoint):
        """
        L63_tlm - function that integrates the Lorenz 1963 equations forward or backward using a TLM and
        its adjoint, given parameters 'par' and initial perturbations 'x0'

        xt = L63_tlm(x0, t, (par, xsave, tsave, adjoint))

         x0 - initial perturbations at time t = 0
          t - vector of time from t = [0, T]
      xsave - states along the control trajectory for the TLM / Adjoint
      tsave - time vector along the control trajectory for the TLM / Adjoint
    adjoint - Forward TLM (False) or Adjoint (True)
         xt - evolved perturbations at time t = T
        """

        s, r, b = self.Par

        x = np.interp(t, tsave, xsave[:, 0])
        y = np.interp(t, tsave, xsave[:, 1])
        z = np.interp(t, tsave, xsave[:, 2])

        M = np.array([[-s, s, 0],
                      [r - z, -1, -x],
                      [y, x, -b]])

        xs = np.dot(np.transpose(M), x0) if adjoint else np.dot(M, x0)

        return xs

    def advance(self, x0, t,
                xtraj=None, ttraj=None, adjoint=False,
                **kwargs):
        """
        method that integrates the L63 model state x0(t0) -> xt(T), or
        TLM (or Adjoint)
        using the appropriate RHS for the L96 system.
        Additional arguments are provided via kwargs.

        xt = advance(self, modelFunc, x0, t, perfect=True, **kwargs)

           x0 - initial state at time t = t0
            t - vector of time from t = [t0, T]
        xtraj - non-linear reference trajectory [DEFAULT: None]
        ttraj - vector of time from t = [t0, T] [DEFAULT: None]
      adjoint - adjoint (True) or forward TLM (False) [DEFAULT: False]
     **kwargs - additional keyword arguments for odeint
           xt - final state at time t = T
        """

        # determine if this is a TLM or Adjoint run
        isTLrun = False if xtraj is None and ttraj is None else True

        if isTLrun:  # Either TLM or Adjoint
            xt = super().advance(self.rhs_tlm, x0, t, (xtraj, ttraj, adjoint), **kwargs)
        else:  # Non-linear run
            xt = super().advance(self.rhs, x0, t, **kwargs)

        return xt

    def plot(
            self,
            obs=None,
            ver=None,
            xb=None,
            xa=None,
            xdim=0,
            ydim=2,
            **kwargs):
        """
        Plot the Lorenz 1963 attractor in 2D

        plot(obs=None, ver=None, xb=None, xa=None, xdim=0, ydim=2, **kwargs)

            obs - x,y,z from t = [0, T]
            ver - x,y,z from t = [0, T]
             xb - prior x,y,z from t = [0, T]
             xa - posterior x,y,z from t = [0, T]
           xdim - variable along x-axis (X)
           ydim - variable along y-axis (Z)
        """

        global xlab
        if xdim == ydim:
            xdim = 0
            ydim = 2

        if xdim < 0 or xdim > 2:
            xdim = 0
        if ydim < 0 or ydim > 2:
            ydim = 2

        if xdim == 0:
            xlab = 'X'
        elif xdim == 1:
            xlab = 'Y'
        elif xdim == 2:
            xlab = 'Z'

        if ydim == 0:
            ylab = 'X'
        elif ydim == 1:
            ylab = 'Y'
        elif ydim == 2:
            ylab = 'Z'

        fig = plt.figure()
        plt.clf()

        att = kwargs.get('att', None)

        if att is not None:
            plt.plot(att[:, xdim], att[:, ydim], color='gray', linewidth=1)
        if xb is not None:
            plt.plot(xb[:, xdim], xb[:, ydim], 'b-', linewidth=1)
        if xa is not None:
            plt.plot(xa[:, xdim], xa[:, ydim], 'r-', linewidth=1)
        # if ver is not None: plt.plot(ver[:,xdim], ver[:,ydim], 'k-', linewidth=1)
        # if obs is not None: plt.plot(obs[:,xdim], obs[:,ydim], 'yo', markeredgecolor='y')

        plt.xlabel(xlab, fontweight='bold', fontsize=12)
        plt.ylabel(ylab, fontweight='bold', fontsize=12)
        title_str = "Lorenz attractor"
        plt.title(title_str, fontweight='bold', fontsize=14)
        fig.canvas.set_window_title(title_str)

        return fig


class L96(LorenzBase):
    """
    This module provides an interface to Lorenz 96 class models
    together with its TL and Adjoint models.
    """

    def __init__(self, modelConfig):
        """
        Populates the L96 class given the model configuration,
        name, time-step, degrees of freedom and other model specific parameters
        """
        super().__init__(modelConfig)

    def rhs(self, x0, t, F, dummy):
        """
        method that describes the RHS of Lorenz and Emanuel 1998 equations,
        given forcing 'F' and initial conditions 'x0'

        xs = L96.rhs(x0, t, (F, dummy))

       x0 - initial state at time t = t0
        t - vector of time from t = [t0, T]
        F - Forcing
    dummy - Arguments coming in after x0, t MUST be a tuple (,) for scipy.integrate.odeint to work
       xs - final state at time t = T
        """

        xs = np.zeros(self.Ndof)

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

    def rhs_tlm(self, x0, t, F, xtraj, ttraj, adjoint):
        """
        rhs_tlm - method that integrates the Lorenz and Emanuel 1998 equations forward or backward
        using a TLM and its adjoint, given Forcing 'F' and initial perturbations 'x0'

        xs = L96.rhs_tlm(x0, t, (F, xsave, tsave, adjoint))

        x0 - initial perturbations at time t = 0
         t - vector of time from t = [0, T]
         F - Forcing
     xtraj - states along the control trajectory for the TLM / Adjoint
     ttraj - time vector along the control trajectory for the TLM / Adjoint
   adjoint - Forward TLM (False) or Adjoint (True)
        xs - evolved perturbations at time t = T
        """

        x = np.zeros(self.Ndof)

        for j in range(self.Ndof):
            x[j] = np.interp(t, ttraj, xtraj[:, j])

        M = np.zeros((self.Ndof, self.Ndof))

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

        xs = np.dot(np.transpose(M), x0) if adjoint else np.dot(M, x0)

        return xs

    def advance(self, x0, t, perfect=True,
                adjoint=False, xtraj=None, ttraj=None,
                **kwargs):
        """
        method that integrates the L96 model state x0(t0) -> xt(T), or
        TLM (or Adjoint)
        using the appropriate RHS for the L96 system.
        Additional arguments are provided via kwargs.

        xt = advance(self, modelFunc, x0, t, perfect=True, **kwargs)

        x0 - initial state at time t = t0
         t - vector of time from t = [t0, T]
   perfect - perfect model [DEFAULT: True]
   adjoint - adjoint (True) or forward TLM (False) [DEFAULT: False]
     xtraj - non-linear reference trajectory [DEFAULT: None]
     ttraj - vector of time from t = [t0, T] [DEFAULT: None]
  **kwargs - any additional keyword arguments
        xt - final state at time t = T
        """

        par = self.Par[0] if perfect else self.Par[1]

        # determine if this is a TLM or Adjoint run
        isTLrun = False if xtraj is None and ttraj is None else True

        if isTLrun:  # Either TLM or Adjoint
            xt = super().advance(self.rhs_tlm, x0, t, (par, xtraj, ttraj, adjoint), **kwargs)
        else:  # Non-linear run
            xt = super().advance(self.rhs, x0, t, (par, 0.0), **kwargs)

        return xt

    def plot(
            self,
            obs=None,
            ver=None,
            xb=None,
            xa=None,
            t=0,
            figNum=None,
            **kwargs):
        """
        Plot the Lorenz 1996 attractor in polar coordinates

        plot(self, obs=None, ver=None, xb=None, xa=None, t=0, N=1, figNum=None, **kwargs)

           obs - observations [None]
           ver - truth [None]
            xb - prior ensemble or ensemble mean [None]
            xa - posterior ensemble or ensemble mean [None]
             t - assimilation time [0]
        figNum - figure handle [None]
        """

        N = self.Ndof

        fig = plt.figure() if figNum is None else plt.figure(figNum)
        plt.clf()
        mean_dist = 35.0
        plt.subplot(111, polar=True)
        theta = np.linspace(0.0, 2 * np.pi, num=N + 1)

        # start by plotting a dummy tiny white dot at (0, 0)
        plt.plot(0, 0, 'w.', markeredgecolor='w', markersize=0.0)

        if xb is not None:
            if len(xb.shape) == 1:
                tmp = np.zeros(N + 1)
                tmp[1:] = xb
                tmp[0] = xb[-1]
            else:
                tmp, tmpmin, tmpmax = self._getLimits(xb)
                plt.fill_between(theta,
                                 tmpmin + mean_dist,
                                 tmpmax + mean_dist,
                                 fc='blue', ec='blue', alpha=0.75)
            plt.plot(theta, tmp + mean_dist, 'b-', linewidth=2.0)

        if xa is not None:
            if len(xa.shape) == 1:
                tmp = np.zeros(N + 1)
                tmp[1:] = xa
                tmp[0] = xa[-1]
            else:
                tmp, tmpmin, tmpmax = self._getLimits(xa)
                plt.fill_between(theta,
                                 tmpmin + mean_dist,
                                 tmpmax + mean_dist,
                                 fc='red', ec='red', alpha=0.5)
            plt.plot(theta, tmp + mean_dist, 'r-', linewidth=2.0)

        if ver is not None:
            tmp = np.zeros(N + 1)
            tmp[1:] = ver
            tmp[0] = ver[-1]
            plt.plot(theta, tmp + mean_dist, 'k-', linewidth=2.0)

        if obs is not None:
            tmp = np.zeros(N + 1)
            tmp[1:] = obs
            tmp[0] = obs[-1]
            plt.plot(theta, tmp + mean_dist, 'yo',
                     markersize=7.5, mec='y', alpha=0.95)

        plt.gca().set_rmin(0.0)
        plt.gca().set_rmax(mean_dist + 25.0)
        rgrid = np.array(np.linspace(10, mean_dist + 25,
                                     num=5, endpoint=False), dtype=int)
        rlabel = []
        rgrid, rlabel = plt.rgrids(rgrid, rlabel)

        tlabel = []
        tgrid = np.array(np.linspace(
            0, 360, num=20, endpoint=False), dtype=int)
        tlabel = np.array(np.linspace(
            0, 40, num=20, endpoint=False), dtype=int)
        tgrid, tlabel = plt.thetagrids(tgrid, tlabel)

        pretitle = None
        for key in kwargs:
            if key == 'pretitle':
                pretitle = kwargs[key]

        title = 'k = %d' % (t) if np.isreal(t) else str(t)
        if pretitle is not None:
            title = pretitle + ' - ' + title
        plt.title(title, fontweight='bold', fontsize=14)
        fig.canvas.set_window_title(title)

        return fig

    def _getLimits(self, xd):
        # {{{
        xmin, xmax, xmean = np.min(xd, axis=1), np.max(
            xd, axis=1), np.mean(xd, axis=1)
        tmpmin = np.zeros(self.Ndof + 1)
        tmpmin[1:] = xmin
        tmpmin[0] = xmin[-1]
        tmpmax = np.zeros(self.Ndof + 1)
        tmpmax[1:] = xmax
        tmpmax[0] = xmax[-1]
        tmp = np.zeros(self.Ndof + 1)
        tmp[1:] = xmean
        tmp[0] = xmean[-1]
        return tmp, tmpmin, tmpmax

    def getIC(self, restart=None, Nens=1):
        """
        Get initial conditions based on model and restart conditions

        [xt, xa] = getIC(restart=False, Nens=None)

         restart - restart dictionary
            Nens - no. of ensemble members [None]
              xt - truth
              xa - analysis or analysis ensemble
        """

        # insure the same sequence of random numbers EVERY TIME
        np.random.seed(0)

        print(('Generating ICs for %s' % self.Name))

        if restart is None:
            print('... from Lorenz and Emanuel, 1998')

            xt = np.ones(self.Ndof) * self.Par[0]
            xt[0] = 1.001 * self.Par[0]

            xa = super().perturb_state(xt, Nens=Nens)

        else:

            [xt, xa] = super().read_from_restart(restart, Nens=Nens)

        return [xt, xa]
