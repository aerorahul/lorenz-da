import numpy as np
from scipy import integrate
from model import AbsModel

class L96(AbsModel):
    def __init__(self, Ndof=40, dt=1.0e-4, F=8.0):
        self.Name = 'L96'
        self.Ndof = Ndof
        self.dt   = dt
        self.Par  = {}
        self.Par['F'] = F  # Forcing term

    def model(x0, t, F, G=None):
        """
        L96 - function that integrates the Lorenz and Emanuel 1998 equations, given forcing 'F', and additional forcing 'G' (eg. IAU tendency) and initial
        conditions 'x0'

        xs = L96(x0, t, (F, G))

       x0 - initial state at time t = 0
        t - vector of time from t = [0, T]
        F - Forcing
        G - Additional forcing (e.g. IAU tendency)
       xs - final state at time t = T
        """

        ndof = len(x0)

        xs = np.zeros(ndof)

        for j in range(ndof):
            jp1 = j + 1
            if (jp1 >= ndof):
                jp1 = jp1 - ndof
            jm2 = j - 2
            if (jm2 < 0):
                jm2 = jm2 + ndof
            jm1 = j - 1
            if (jm1 < 0):
                jm1 = jm1 + ndof

            xs[j] = (x0[jp1] - x0[jm2]) * x0[jm1] - x0[j] + F
            if G is not None:
                xs[j] = xs[j] + G[j]

        return xs


    def model_tlm(x0, t, F, xsave, tsave, adjoint):
        """
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
        """

        ndof = len(x0)
        x = np.zeros(ndof)

        for j in range(ndof):
            x[j] = np.interp(t, tsave, xsave[:, j])

        M = np.zeros((ndof, ndof))

        for j in range(ndof):
            jp1 = j + 1
            if (jp1 >= ndof):
                jp1 = jp1 - ndof
            jm2 = j - 2
            if (jm2 < 0):
                jm2 = jm2 + ndof
            jm1 = j - 1
            if (jm1 < 0):
                jm1 = jm1 + ndof
            M[j, jm2] = -x[jm1]
            M[j, jm1] = x[jp1] - x[jm2]
            M[j, j] = -1
            M[j, jp1] = x[jm1]

        if (adjoint):
            xs = np.dot(np.transpose(M), x0)
        else:
            xs = np.dot(M, x0)

        return xs

    def advance(self, x0, t, perfect=True, result=None, **kwargs):
        """
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
        """

        func = self.model
        if (perfect):
            par = self.Par[0]
        else:
            par = self.Par[1]

        # exec('xs = integrate.odeint(self.%s, x0, t, (par, 0.0), **kwargs)' % (self.Name))
        xs = integrate.odeint(func, x0, t, (par, 0.0), **kwargs)

        if (result is None):
            return xs
        else:
            result.put(xs)

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
        """
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
        """

        func = self.model_tlm
        if (perfect):
            par = self.Par[0]
        else:
            par = self.Par[1]

        if (adjoint):
            xref = np.flipud(xref)

        xs = integrate.odeint(
            func, x0, t, (par, xref, tref, adjoint), **kwargs)

        if (result is None):
            return xs
        else:
            result.put(xs)