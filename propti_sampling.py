import sys
import os
import typing
import numpy as np
import copy
import pandas as pd
import shutil as sh
import scipy

import propti as pr

import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str,
                    help="python input file containing parameters and "
                         "simulation setups")
parser.add_argument("--root_dir", type=str,
                    help="root directory for sampling process", default='.')
cmdl_args = parser.parse_args()

setups = None  # type: pr.SimulationSetupSet
params = None  # type: pr.ParameterSet
sampler = None  # type: pr.Sampler

input_file = cmdl_args.input_file

logging.info("reading input file: {}".format(input_file))
exec(open(input_file).read(), globals())
 
# TODO: check for correct execution
if params is None:
    logging.critical("sampling parameters not defined")
if setups is None:
    logging.critical("simulation setups not defined")
if sampler is None:
    logging.critical("sampler properties not defined")

input_file_directory = os.path.dirname(input_file)
logging.info("input file directory: {}".format(input_file_directory))


# TODO: put the following lines into a general function (basic_functions.py)?
for s in setups:

    cdir = os.path.join(cmdl_args.root_dir, s.work_dir)

    # create work directories
    if not os.path.exists(cdir):
        os.mkdir(cdir)

    # copy model template
    sh.copy(os.path.join(input_file_directory, s.model_template), cdir)

    s.model_template = os.path.join(cdir, os.path.basename(s.model_template))

    # TODO Sampler: add comparison capability 
    # # copy all experimental data
    # # TODO: Re-think the copy behaviour. If file is identical, just keep one
    # # instance?
    # for r in s.relations:
    #     if r.experiment is not None:
    #         sh.copy(os.path.join(input_file_directory, r.experiment.file_name), cdir)
    #         r.experiment.file_name = os.path.join(cdir, os.path.basename(r.experiment.file_name))

# check for potential non-unique model input files
in_file_list = []
for s in setups:
    tpath = os.path.join(s.work_dir, s.model_input_file)
    logging.debug("check if {} is in {}".format(tpath, in_file_list))
    if tpath in in_file_list:
        logging.error("non unique module input file path: {}".format(tpath))
        sys.exit()
    in_file_list.append(tpath)

logging.info(setups)
logging.info(params)
logging.info(sampler)

for s in setups:
    os.mkdir(s.execution_dir_prefix)
    sample_set = sampler.create_sample_set(s.model_parameter)

    para_table_file = open(os.path.join(s.execution_dir_prefix, 'sample_table.csv'), 'w')

    line_name = "# NAME  -  index"
    line_min = f"# MIN   - {0:6d}"
    line_max = f"# MAX   - {sampler.nsamples-1:6d}"
    lines_consts = ""
    next_tabs = 1
    for p in s.model_parameter:
        if p.max_value is None or p.min_value is None:
            lines_consts += f"# CONST - {p.name} = {p.value}\n"
            continue
        line_name += ','+'\t'*next_tabs + p.name
        line_min += f",\t{p.min_value:.6e}"
        line_max += f",\t{p.max_value:.6e}"

        next_tabs = len(p.name) // 8
        
    para_table_file.write(lines_consts)
    para_table_file.write(line_name + '\n')
    para_table_file.write(line_min + '\n')
    para_table_file.write(line_max + '\n')

    logging.debug("Sample set:\n")
    sample_index = 0
    for sample in sample_set:
        logging.debug(f"  {sample}")

        line = f"\t\t{sample_index:6d}"
        for p in sample:
            if p.max_value is None or p.min_value is None:
                continue
            line += ',\t' + f"{p.value:.6e}"
        para_table_file.write(line + '\n')

        tmp_simulation_setup = typing.cast(pr.SimulationSetup, copy.deepcopy(s))
        tmp_simulation_setup.model_parameter = sample
        tmp_simulation_setup.execution_dir = sample.name
        os.mkdir(os.path.join(tmp_simulation_setup.execution_dir_prefix, tmp_simulation_setup.execution_dir))
        pr.create_input_file(tmp_simulation_setup)

        sample_index += 1

    para_table_file.close()

# if cmdl_args.prepare_init_inputs:
#     logging.info("prepare input files with initial values")
#     for s in setups:
#         pr.create_input_file(s)
