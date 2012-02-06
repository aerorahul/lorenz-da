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

def L63(x0, t, par, dummy):
# {{{
    '''
    L63 - function that integrates the Lorenz 1963 equations, given parameters 'par' and initial
    conditions 'x0'

    xs = L63(x0, t, (par, dummy))

       xs - final state at time t = T
       x0 - initial state at time t = 0
        t - vector of time from t = 0 to t = T
      par - parameters of the Lorenz system
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
    its adjoint, given parameters 'par' and initial conditions 'x0'

    xs = L63_tlm(x0, t, (par, xsave, tsave, adjoint))

         xs - final state at time t = T
         x0 - initial state at time t = 0
          t - vector of time from t = 0 to t = T
        par - parameters of the Lorenz system
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
