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
setups, ops, optimiser = pickle.load(in_file)
in_file.close()
# Check if everything is in order
if ops is None:
    logging.critical("optimisation parameter are not defined")
if setups is None:
    logging.critical("simulation setups are not defined")
if optimiser is None:
    logging.critical("optimiser properties are not defined")

# Optimiser to perform FAST sensitivity analysis.

# Use the same data from optimiser for sensitivity analysis
num_subproc = optimiser.num_subprocesses
mpi_bool = optimiser.mpi
backup = optimiser.backup_every

# Compute number of repetitions required
k = len(ops)        # total number of optimization params
(M, d) = 3, 2       # M = inference factor, d = freq. step <spotpy defaults>
rep = (1 + 4*(1 + (k-2)*d)*M**2)*k


sensitivity = pr.OptimiserProperties('fast',
                                     repetitions=rep,
                                     backup_every=backup,
                                     db_name="propti_sensitivity_db",
                                     db_type="csv",
                                     num_subprocesses=num_subproc,
                                     mpi=mpi_bool)


print(setups, ops, sensitivity)
'''
Not writing any data to pickle file since there is only
one method to analyse sensitivity. It makes sense to Wiki this instead.
'''
# Write sensitivity pickle file
# out_file = open('propti_sensitivity.pickle.init', 'wb')
# pickle.dump((setups, ops, optimiser), out_file)
# out_file.close()

# Run optimization.
# Output of run_optimisation will be None.
# What are some good parameters that can be printed ?
res = pr.run_optimisation(ops, setups, sensitivity)
# print(res) # output of res is None.
