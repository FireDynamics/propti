import numpy as np
import propti as pr

import pickle

import propti as pr

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str,
                    help="optimisation root directory")
cmdl_args = parser.parse_args()

setups = None  # type: pr.SimulationSetupSet
ops = None  # type: pr.ParameterSet
optimiser = None  # type: pr.OptimiserProperties

in_file = open('propti.pickle.init', 'rb')
setups, ops, *rest = pickle.load(in_file)
in_file.close()
# set optimiser to perform FAST sensitivity analysis.
# This will not overwrite pickle file, but create a new one.
k = len(ops)        # total number of optimization params
(M, d) = 3, 2       # M = inference factor, d = freq. step <spotpy defaults>
rep = (1 + 4*(1 + (k-2)*d)*M**2)*k
optimiser = pr.OptimiserProperties('fast', repetitions=rep, mpi=False)

if ops is None:
    logging.critical("optimisation parameter are not defined")
if setups is None:
    logging.critical("simulation setups are not defined")

print(setups, ops, optimiser)
# Run optimization.
# Output of run_optimisation will be None.
# What are some good parameters that can be printed ?
res = pr.run_optimisation(ops, setups, optimiser)
# print(res) # output of res is None.
