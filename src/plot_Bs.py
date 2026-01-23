#!/usr/bin/env python

###############################################################
# plot_Bs.py - plot the raw static covariance matrix
###############################################################

###############################################################
from plot_stats import plot_cov
from module_IO import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot
import sys
###############################################################

###############################################################
###############################################################

parser = ArgumentParser(
    description='Plot climatological covariances for LXX models',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m',
    '--model',
    help='model name',
    type=str,
    choices=[
        'L63',
        'L96'],
    default='L96',
    required=False)
parser.add_argument(
    '-f',
    '--filename',
    help='file with climatological covariances',
    type=str,
    required=True)
args = parser.parse_args()

model = type('', (), {})
model.Name = args.model

Bs = read_clim_cov(fname=args.filename)

fig = plot_cov(Bs, title="Climatological : $\\mathbf{B}_c$")

pyplot.show()
sys.exit(0)
