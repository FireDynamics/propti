import sys
import os
import numpy as np
import copy
import pandas as pd
import shutil as sh
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
ver, setups, ops, optimiser = pickle.load(in_file)
in_file.close()

if ver.flag_propti != 0:
    logging.warning("No git. Propti version is represented as a hash.")
if ops is None:
    logging.critical("optimisation parameter are not defined")
if setups is None:
    logging.critical("simulation setups are not defined")
if optimiser is None:
    logging.critical("optimiser properties are not defined")

print(ver, setups, ops, optimiser)

res = pr.run_optimisation(ops, setups, optimiser)

print(ops)

out_file = open('propti.pickle.finished', 'wb')
pickle.dump((ver, setups, ops, optimiser), out_file)
out_file.close()
