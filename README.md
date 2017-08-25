lorenz-da
========

The purpose of this project is to test different data assimilation (DA) algorithms, ensemble, variational and hybrid, on simple Lorenz models.
  
This project is actively being worked on and is hosted on GitHub for educational purposes only. 

Please report bugs or suggestions for improvements. 


 <i class="icon-file"></i>**module_Lorenz.py** - contains Lorenz model related functions; e.g. model non-linear integration,  tangent-linear integration and adjoint, plotting model attractor, generating initial conditions, etc.

<i class="icon-file"></i>**module_DA.py** - contains data assimilation related functions; e.g. variational DA with 3DVar, 4DVar, ensemble DA with stochastic perturbed observations EnKF, square-root filter with modified Kalman gain EnKF,  ensemble adjustment Kalman filter, variety of options for inflation, localization, minimization, etc.

<i class="icon-file"></i>**module_IO.py** - utility functions for IO; e.g. create, read and write model and data assimilation diagnostic files.

<i class="icon-file"></i>**LXX_model.py** - drive a standalone Lorenz 63 (3 variable )or Lorenz 96 (40 variable) model along with its tangent-linear and adjoint. Also check for accuracy of the TLM and ADJ.

<i class="icon-file"></i>**truthDA.py** - driver script to generate truth for DA.
<i class="icon-file"></i>**varDA.py** - driver script for variational DA
<i class="icon-file"></i>**ensDA.py** - driver script for ensemble DA
<i class="icon-file"></i>**hybDA.py** - driver script for hybrid DA
<i class="icon-file"></i>**hybensvarDA.py** - driver script for hybrid ensemble-variational DA

<i class="icon-file"></i>**LXX_stats.py** - Variational DA requires a static background error covariance matrix. 3 methods are supported here; a long free integration of the model to sub-sample states from, NMC method using lagged forecast pairs, and using an ensemble from a cycled EnKF system.

<i class="icon-file"></i>**L63_param_truthDA** - parameters to generate the truth
<i class="icon-file"></i>**L63_param_varDA** - parameters to configure variational DA setup
<i class="icon-file"></i>**L63_param_ensDA** - paramaters to configure ensemble DA setup
<i class="icon-file"></i>**L63_param_hybDA** - parameters to configure hybrid DA setup

<i class="icon-file"></i>**L96_param_truthDA** - parameters to generate the truth
<i class="icon-file"></i>**L96_param_varDA**- parameters to configure variational DA setup
<i class="icon-file"></i>**L96_param_ensDA** - paramaters to configure ensemble DA setup
<i class="icon-file"></i>**L96_param_hybDA** - parameters to configure hybrid DA setup
<i class="icon-file"></i>**L96_param_ensvarDA** - parameters to configure pure ensemble-variational DA
<i class="icon-file"></i>**L96_param_hybensvarDA** - parameters to configure hybrid ensemble-variational DA

<i class="icon-file"></i>**comp_varDA.py** - make comparison plots between different configurations of variational DA runs
<i class="icon-file"></i>**comp_ensDA.py** - make comparison plots between different configurations of ensemble DA runs
<i class="icon-file"></i>**comp_hybDA.py** - make comparison plots between different configurations of hybrid DA runs.

<i class="icon-file"></i>**plot_Bs.py** - read and plot background error covariance
<i class="icon-file"></i>**test_localization.py** - read, localize and plot background error covariance. Localization options are Gaspari-Cohn, Buehner, and Liu.
<i class="icon-file"></i>**loop_LXX.py** - read diagnostic files and make animated loops of truth, prior, posterior (including ensemble) and observations.

Several more scripts to do forecast sensitivity to observations and observation impacts are included here, and their description will be updated at a later time.
