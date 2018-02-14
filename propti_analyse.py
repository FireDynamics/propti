import os
import pandas as pd
import pickle
import logging
import argparse
import sys

import propti.basic_functions as pbf

# import matplotlib.pyplot as plt

import propti as pr
import propti.propti_monitor as pm
import propti.propti_post_processing as ppm

# TODO: Create subdirectory for process analysis and plots automatically.

parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str,
                    help="optimisation root directory",
                    default='.')

parser.add_argument("--create_best_input",
                    help="Creates simulation input file  with "
                         "best parameter set",
                    action="store_true")

parser.add_argument("--run_best",
                    help="run simulation(s) with best parameter set",
                    action="store_true")

parser.add_argument("--plot_fitness_development",
                    help="Scatter plot of fitness values", action="store_true")

parser.add_argument("--plot_para_values",
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


def check_directory(dir_list):
    """
    Take a list of directory names (strings) and attach them to the root
    path. Check if this path exists, if not create it.
    :param dir_list: List containing the directory names, as string.
    :return: New file path, based on files root and user input.
    """

    # Set up new path.
    new_dir = os.path.join(cmdl_args.root_dir)
    for i in dir_list:
        new_dir = os.path.join(new_dir, i)

    # Check if the new path exists, otherwise create it.
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Return new path for further usage.
    return new_dir


print("")
print("* Loading information of the optimisation process.")
print("----------------------")

# Check if `propti.pickle.finish` exists, else use `propti.pickle.init`.
# if os.path.isfile(os.path.join(cmdl_args.root_dir, 'propti.pickle.finished')):
#     pickle_file = os.path.join(cmdl_args.root_dir, 'propti.pickle.finished')
# elif os.path.isfile(os.path.join(cmdl_args.root_dir, 'propti.pickle.init')):
#     pickle_file = os.path.join(cmdl_args.root_dir, 'propti.pickle.init')
# else:
#     sys.exit("Neither 'propti.pickle.finished' nor 'propti.pickle.init' "
#              "detected. Script execution stopped.")

pickle_file = os.path.join(cmdl_args.root_dir, 'propti.pickle.init')

in_file = open(pickle_file, 'rb')
setups, ops, optimiser = pickle.load(in_file)
in_file.close()

print("Loading complete.")


if setups is None:
    logging.critical("simulation setups are not defined")

if ops is None:
    logging.critical("optimisation parameter are not defined")

# print(setups, ops, optimiser)

# TODO: define spotpy db file name in optimiser properties
# TODO: use placeholder as name? or other way round?


if cmdl_args.run_best:
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    print("")
    print("- run simulation(s) of best parameter set")
    print("----------------------")
    pr.run_best_para(setups, ops, optimiser, pickle_file)
    print("")
    print("")


if cmdl_args.create_best_input:
    """
    Takes the (up to now) best parameter set from the optimiser data base and 
    reads the corresponding parameter values. The parameter values are written 
    into the simulation input file and saved as *_bestpara.file-type.
    This functionality is focused on the usage of SPOTPY.
    """

    print("")
    print("* Create input file with best parameter set")
    print("----------------------")
    print("Read data base file, please wait.")
    print("")

    # Read data base file name from the pickle file.
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'BestParameter'])

    # Collect the parameter names. Change format to match column headers, based
    # on SPOTPY definition. Store headers in a list.
    cols = []
    for p in ops:
        cols.append("par{}".format(p.place_holder))

    # Determine the best fitness value and its position.
    print("Locate best parameter set:")
    print("---")
    fitness_values = pd.read_csv(db_file_name, usecols=['like1'])
    best_fitness_index = fitness_values.idxmax().iloc[0]
    best_fitness_value = fitness_values.max().iloc[0]

    print("Best fitness index: line {}".format(best_fitness_index))
    print("Best fitness value: {}".format(best_fitness_value))
    print("")

    # Load template.
    template_file_path = setups[0].model_template
    print(template_file_path)
    temp_raw = pbf.read_template(template_file_path)
    print(temp_raw)

    # Extract the parameter values of the best set.
    print("* Extract best parameter set")
    print("----------------------")
    print("Read data base file, please wait.")
    print("")
    print("Parameters:")
    print("---")
    parameter_values = pd.read_csv(db_file_name, usecols=cols)

    for i in range(len(cols)):
        new_para_value = parameter_values.at[best_fitness_index, cols[i]]
        print("{}: {}".format(cols[i][3:], new_para_value))

        if type(new_para_value) == float:
            temp_raw = temp_raw.replace("#" + cols[i][3:] + "#",
                              "{:E}".format(new_para_value))
        else:
            temp_raw = temp_raw.replace("#" + cols[i][3:] + "#", str(new_para_value))


    new_path = os.path.join(results_dir, 'best_para.fds')

    pbf.write_input_file(temp_raw, new_path)
    print("")
    print("Simulation input file with best parameter set written.")

    print("Task finished.")
    print("")
    print("")


if cmdl_args.plot_fitness_development:

    """
    Scatter plot of fitness value (RMSE) development. It reads the propti data 
    base file, based on information stored in the pickle file. 
    This functionality is focused on the usage of SPOTPY.
    """

    print("")
    print("* Plot fitness values.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'Plots', 'Scatter'])

    # Extract data to be plotted.
    cols = ['like1']
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    pr.plot_scatter(cols[0], data,
                    'Fitness value development', 'FitnessDevelopment',
                    results_dir, 'Root Mean Square Error (RMSE)')

    print("Plot(s) have been created.")
    print("")
    print("")


# Scatter plot of RMSE development
if cmdl_args.plot_para_values:
    """
    Creates scatter plots of the development of each parameter over the 
    optimisation process. It reads the propti data 
    base file, based on information stored in the pickle file. 
    This functionality is focused on the usage of SPOTPY.
    """

    print("")
    print("* Plot likes and values.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir_scatter = check_directory(['Analysis', 'Plots', 'Scatter'])
    results_dir_boxplot = check_directory(['Analysis', 'Plots', 'Boxplot'])
    results_dir_histogram = check_directory(['Analysis', 'Plots', 'Histogram'])

    # Extract data to be plotted.
    cols = ['like1', 'chain']
    for p in ops:
        cols.append("par{}".format(p.place_holder))
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    for c in cols[2:]:
        # Scatter plots of parameter development over the whole run.
        pr.plot_scatter(c, data, 'Parameter development: ' + c, c,
                        results_dir_scatter)

        # Histogram plots of parameters
        pr.plot_hist(c, data, 'Histogram per generation for: ' + c,
                     c, results_dir_histogram, y_label=None)

    # Scatter plot of fitness values.
    pr.plot_scatter('like1', data, 'Fitness value development',
                    'FitnessDevelopment', results_dir_scatter,
                    'Root Mean Square Error (RMSE)')

    # Box plot to visualise steps (generations).
    pr.plot_box_rmse(data, 'Fitness values, histogram per step (generation)',
                     len(ops),
                     optimiser.ngs,
                     'FitnessDevelopment', results_dir_boxplot)

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
        pr.plot_best_sim_exp(s, pickle_file)
    print("")
    print("")
