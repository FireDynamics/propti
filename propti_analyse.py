import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import propti as pr
import propti.propti_monitor as pm
import propti.propti_post_processing as ppm
import logging

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str,
                    help="optimisation root directory")

parser.add_argument("--run_best",
                    help="run simulation(s) with best parameter set",
                    action="store_true")

parser.add_argument("--plot_like_values",
                    help="plot like and values", action="store_true")

parser.add_argument("--calc_stat",
                    help="calculate statistics", action="store_true")

parser.add_argument("--plot_best_sim_exp",
                    help="plot results of the simulation of the best parameter "
                         "set and the experimental data to be compared with",
                    action="store_true")
cmdl_args = parser.parse_args()

setups = None  # type: pr.SimulationSetupSet
ops = None  # type: pr.ParameterSet
optimiser = None  # type: pr.OptimiserProperties

pickle_finished = os.path.join(cmdl_args.root_dir, 'propti.pickle.finished')

in_file = open(pickle_finished, 'rb')
ver, setups, ops, optimiser = pickle.load(in_file)
in_file.close()

if setups is None:
    logging.critical("simulation setups are not defined")

if ops is None:
    logging.critical("optimisation parameter are not defined")

print(ver, setups, ops, optimiser)

# TODO: define spotpy db file name in optimiser properties
# TODO: use placeholder as name? or other way round?


if cmdl_args.run_best:
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    print("")
    print("- run simulation(s) of best parameter set")
    print("----------------------")
    pr.run_best_para(setups, ops, optimiser, pickle_finished)
    print("")
    print("")


# Scatter plot of RMSE development
if cmdl_args.plot_like_values:
    print("")
    print("- plot likes and values")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Extract data to be plotted.
    cols = ['like1', 'chain']
    for p in ops:
        cols.append("par{}".format(p.place_holder))
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    for c in cols[2:]:
        pr.plot_scatter(c, data, 'Parameter development', file_name=c, plot_text=ver)

    # Histogram plots of parameters
    for c in cols[2:]:
        pr.plot_hist(c, data, 'histogram', y_label=None)
    pr.plot_scatter('like1', data, 'RMSE', 'Fitness values',
                    'Root Mean Square Error (RMSE)')

    # Box plot to visualise steps (generations).
    pr.plot_box_rmse(data, 'RMSE', len(ops), optimiser.ngs, 'Fitness values')

    print("Plots have been created.")
    print("")
    print("")


if cmdl_args.calc_stat:
    # TODO: write statistics data to file

    print("")
    print("- calculate statistics")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    for s in setups:
        cols = []
        lab = ['like1']
        for p in ops:
            cols.append("par{}".format(p.place_holder))
            lab.append("par{}".format(p.place_holder))

        data_raw = pd.read_csv(db_file_name, usecols=cols)

        data = []
        for i in cols:
            data.append(data_raw[i])

        fname = s.analyser_input_file
        with open(fname) as f:
            content = f.readlines()

        for line in content:
            if 'pearson_coeff' in line:
                pear_coeff = True

    if pear_coeff is True:
        print('Pearson coefficient matrix for the whole run:')
        mat = pr.calc_pearson_coefficient(data)
        print('')

    data_fit = pd.read_csv(db_file_name, usecols=lab)
    # print(data_fit.head())
    # print('')
    data_fit['like1'].tolist()
    t = pr.collect_best_para_multi(db_file_name, lab)
    # print(t)
    print('')

    best_para_sets = []
    for i in cols:
        best_para_sets.append(t[i])

    print('Pearson coefficient matrix for the best parameter collection:')
    mat_best_collection = pr.calc_pearson_coefficient(best_para_sets)
    print('')


if cmdl_args.plot_best_sim_exp:
    # TODO: write statistics data to file

    print("")
    print("- plot best simulation and experimental data")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    for s in setups:
        pr.plot_best_sim_exp(s, pickle_finished)
    print("")
    print("")

