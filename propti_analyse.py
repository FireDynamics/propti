#!/usr/bin/env python3
import os
import pandas as pd
import pickle
import logging
import argparse
import sys
import copy
import shutil

import propti.basic_functions as pbf

# import matplotlib.pyplot as plt

import propti as pr
import propti.propti_monitor as pm
import propti.propti_post_processing as ppm


parser = argparse.ArgumentParser()
parser.add_argument("root_dir", type=str,
                    help="optimisation root directory (location of the 'propti_db.csv', e.g. '.')",
                    default='.')

parser.add_argument("--inspect_init",
                    help="provide overview over the data stored in the "
                         "'pickle.init' file",
                    action="store_true")

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

parser.add_argument("--dump_plots",
                    help="plot like and values", action="store_true")

parser.add_argument("--calc_stat",
                    help="calculate statistics", action="store_true")

parser.add_argument("--plot_best_sim_exp",
                    help="plot results of the simulation of the best parameter "
                         "set and the experimental data to be compared with",
                    action="store_true")

parser.add_argument("--plot_best_para_gen",
                    help="Plots the value of the best parameter set for each"
                         "parameter, by generation.",
                    action="store_true")

parser.add_argument("--plot_fit_semilogx",
                    help="Plots fitness values with semi-log x scale.",
                    action="store_true")

parser.add_argument("--extract_data",
                    help="Extracts parameter data, based on fitness values.",
                    action="store_true")

parser.add_argument("--extractor_sim_input",
                    help="Creates input files, based on  the resulting file "
                         "from the data extractor.",
                    action="store_true")

parser.add_argument("--create_case_input",
                    help="Creates input files for user cases, based on  the "
                         "resulting file from the data extractor.",
                    action="store_true")

parser.add_argument("--clean_db",
                    help="Removes restart markers from the database file.",
                    action="store_true")

parser.add_argument("--func_test",
                    help="Executes test function for testing purpose",
                    action="store_true")
parser.add_argument("--plot_para_vs_fitness",
                    help="Plots each parameter against the fitness values, "
                         "colour coded by repetition.",
                    action="store_true")
# Prototyping of ne analysis script.
parser.add_argument("--create_new_database",
                    help="Creates a new database file from the spotpy data "
                         "base CSV (propti_db.csv)",
                    action="store_true")

cmdl_args = parser.parse_args()

ver = None  # type: pr.Version
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
    for directory in dir_list:
        new_dir = os.path.join(new_dir, directory)

    # Check if the new path exists, otherwise create it.
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Return new path for further usage.
    return new_dir


# Names of sub-directories that are used to contain the results of the
# analysis.
p1, p2, p3 = 'Analysis', 'Plots', 'Extractor'


print("")
print("* Loading information of the optimisation process.")
print("----------------------")

# Check if `propti.pickle.finish` exists, else use `propti.pickle.init`.
if os.path.isfile(os.path.join(cmdl_args.root_dir, 'propti.pickle.finished')):
     pickle_file = os.path.join(cmdl_args.root_dir, 'propti.pickle.finished')
elif os.path.isfile(os.path.join(cmdl_args.root_dir, 'propti.pickle.init')):
     pickle_file = os.path.join(cmdl_args.root_dir, 'propti.pickle.init')
else:
     sys.exit("Neither 'propti.pickle.finished' nor 'propti.pickle.init' "
              "detected. Script execution stopped.")

in_file = open(pickle_file, 'rb')

#######################################################
# TODO: Enable better backwards compatibility than the following:

pickle_items = []
for item in pickle.load(in_file):
    pickle_items.append(item)

in_file.close()

p_length = len(pickle_items)

print('Pickle length: {}'.format(p_length))

if p_length == 3:
    setups, ops, optimiser = pickle_items
elif p_length == 4:
    ver, setups, ops, optimiser = pickle_items
else:
    print('The init-file is incompatible '
          'with this version of propti_analyse.')
#
#######################################################


print("Loading complete.")

# Check if all components are there, otherwise write message to the log file.
if ver is None:
    logging.critical("* Version(s) not defined. Legacy '*.pickle.init' file?")

if setups is None:
    logging.critical("* Simulation setups are not defined.")

if ops is None:
    logging.critical("* Optimisation parameters are not defined.")

if optimiser is None:
    logging.critical("* Optimiser parameters are not defined.")


# TODO: define spotpy db file name in optimiser properties
# TODO: use placeholder as name? or other way round?


##########################
# Inspect PROPTI Init File
if cmdl_args.inspect_init:
    """
    Calls the various print methods of the respective PROPTI objects and 
    prints their content in human-readable form.
    Used to check how the IMP is set up (content of the 'propti.pickle.init'). 
    """

    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    print("")
    print("* Inspection of the 'pickle.init' content")
    print("----------------------")

    print("* Version(s):")
    print(ver)

    print("* Simulation Setups:")
    print(setups)

    print("* Optimisation Parameters:")
    print(ops)

    print("* Optimiser Settings:")
    print(optimiser)

    print("")
    print("")


######################################
# Run Simulation of Best Parameter Set
if cmdl_args.run_best:
    """
    Extracts the best parameter set from the data base and writes it into a 
    copy of the simulation input template. Afterwards, the simulation is 
    executed. 
    """

    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))
    
    
    # Check if a directory for the result files exists. If not, create it.
    results_dir = check_directory([p1, 'RunBestPara'])


    print("")
    print("* Run simulation(s) of best parameter set")
    print("----------------------")
    pr.run_best_para(setups, ops, optimiser, pickle_file)
    print("")
    print("")


###################
# Create Best Input
if cmdl_args.create_best_input:
    """
    Takes the (up to now) best parameter set from the optimiser data base and 
    reads the corresponding parameter values. The parameter values are written 
    into the simulation input file and saved as `*_bestpara.file-type`.
    This functionality is focused on the usage of SPOTPY.
    """

    print("")
    print("* Create input file with best parameter set")
    print("----------------------")
    print("Reading data base file, please wait...")
    print("")

    # Read data base file name from the pickle file.
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Determine the best fitness value and its location in the data base.
    print("* Locate best parameter set:")
    print("---")

    fitness_values = pd.read_csv(db_file_name, usecols=['like1'])
    best_fitness_index = fitness_values.idxmin().iloc[0]
    best_fitness_value = fitness_values.min().iloc[0]

    print("Best fitness index: line {}".format(best_fitness_index))
    print("Best fitness value: {}".format(best_fitness_value))
    print("---")
    print("")

    # Check if a directory for the result files exists. If not, create it.
    results_dir = check_directory([p1, p3, 'CurrentBestParameter',
                                   'Repetition_{}'.format(best_fitness_index)])

    # Collect simulation setup names.
    print("* Collect simulation setup names:")
    print("---")

    sim_setup_names = []
    for ssn in range(len(setups)):
        ssn_value = setups[ssn].name
        sim_setup_names.append(ssn_value)

        print("Setup {}: {}".format(ssn, ssn_value))
    print("---")
    print("")

    # Collect the optimisation parameter names. Change format to match column
    # headers in propti_db, based on SPOTPY definition. Store headers in a list.
    cols = []
    for p in ops:
        cols.append("par{}".format(p.place_holder))

    # Collect parameter names.
    print("* Collect parameter names and place holders:")
    print("---")

    para_names = []
    para_simsetup_complete = []
    para_name_list = []
    for s_i in range(len(setups)):

        # Place holder list.
        para_ph_list = []

        # Collect meta parameters, those which describe the simulation setup.
        # First, find all parameters and place holders.
        para_meta_simsetup = []
        for s_j in range(len(setups[s_i].model_parameter.parameters)):
            paras = setups[s_i].model_parameter.parameters

            # Parameter names.
            para_name = paras[s_j].name
            para_name_list.append(para_name)

            # Place holders.
            para_ph = paras[s_j].place_holder
            para_ph_list.append(para_ph)

            # Compare the place holders with the optimisation parameters, to
            # determine if they are meta parameters.
            p_i = 'par{}'.format(para_ph)
            if p_i not in cols:
                # Store meta parameters (place holder and value) in list.
                para_meta_simsetup.append([para_ph, paras[s_j].value])

            print('Name: {}'.format(para_name))
            print('Place holder: {}'.format(para_ph))
        print("---")

        # Put meta lists into list which mirrors the simulation setups.
        para_simsetup_complete.append(para_meta_simsetup)

    print("")

    print("* Extract best parameter set")
    print("---")
    print("Read data base file, please wait...")
    print("")

    # Read PROPTI data base.
    parameter_values = pd.read_csv(db_file_name, usecols=cols)

    print("Best parameter values:")
    print("---")

    # Extract the parameter values of the best set. Store place holder and
    # parameter values in lists.
    opti_para = []
    for i in range(len(cols)):
        new_para_value = parameter_values.at[best_fitness_index, cols[i]]
        print("{}: {}".format(para_name_list[i], new_para_value))
        opti_para.append([cols[i][3:], new_para_value])

    # Append optimisation parameter place holders and values to the parameter
    # lists, sorted by simulation setups.
    for pssc in para_simsetup_complete:
        for para in opti_para:
            pssc.append(para)

    # print("para complete: {}".format(para_simsetup_complete))
    print("")

    # Load templates from each simulation setup, fill in the values and write
    # the new input files in the appropriate directories.
    print("* Fill templates")
    print("--------------")
    # Counter of simulation setups.
    css = 0
    for simsetup in sim_setup_names:
        # Create new directories, based on simulation setup names.
        check_directory([results_dir, simsetup])

        # Load template.
        template_file_path = setups[css].model_template
        temp_raw = pbf.read_template(template_file_path)

        # Create new input files with best parameters,
        # based on simulation setups.
        for bestpara in para_simsetup_complete[css]:
            print("Best para: {}".format(bestpara))
            new_para_value = bestpara[1]

            # Account for scientific notation of floats.
            if type(new_para_value) == float:
                temp_raw = temp_raw.replace("#" + bestpara[0] + "#",
                                            "{:E}".format(new_para_value))
            else:
                temp_raw = temp_raw.replace("#" + bestpara[0] + "#",
                                            str(new_para_value))

        # Write new input file with best parameters.
        bip = os.path.join(results_dir, simsetup,
                           simsetup + '_rep{}.fds'.format(best_fitness_index))
        pbf.write_input_file(temp_raw, bip)

        print("---")
        # Advance counter.
        css += 1

    print("")
    print("Simulation input file, based on best parameter set, was written.")

    print("* Task finished.")
    print("")
    print("")


##########################
# Plot Fitness Development
if cmdl_args.plot_fitness_development:
    """
    Scatter plot of fitness value (RMSE) development. It reads the PROPTI data 
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


####################################
# Plot Development of All Parameters
if cmdl_args.plot_para_values:
    """
    This functionality is deprecated!
    
    Creates scatter plots of the development of each parameter over the 
    optimisation process. It reads the propti data 
    base file, based on information stored in the pickle file. 
    This functionality is focused on the usage of SPOTPY.
    """

    # TODO: Check for optimisation algorithm
    # TODO: Adjust output depending on optimisation algorithm

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
    results_dir_para_gen = check_directory(['Analysis', 'Plots', 'Para_Gen'])
    results_dir_log = check_directory(['Analysis', 'Plots', 'Log'])

    # Extract data to be plotted.
    cols = ['like1', 'chain']
    for p in ops:
        cols.append("par{}".format(p.place_holder))
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    for c in cols[2:]:
        # Scatter plots of parameter development over the whole run.
        pr.plot_scatter(c, data, 'Parameter development: ', c,
                        results_dir_scatter, version=ver.ver_propti)

        # Histogram plots of parameters
        pr.plot_hist(c, data, 'Histogram per generation for: ' + c,
                     c, results_dir_histogram, y_label=None)

    # Scatter plot of fitness values.
    pr.plot_scatter('like1', data, 'Fitness value development',
                    'FitnessDevelopment', results_dir_scatter,
                    'Root Mean Square Error (RMSE)',
                    version=ver.ver_propti)

    # Plot values of best parameter set, by generation.
    pm.plot_best_para_generation(cols, data, len(ops), optimiser.ngs,
                                 results_dir_para_gen)

    # Plot fitness semi-log x.
    pm.plot_semilogx_scatter('like1', data, 'Fitness value development',
                             'FitnessDevelopment', results_dir_log,
                             'Root Mean Square Error (RMSE)',
                             version=ver.ver_propti)

    # Box plot to visualise steps (generations).
    pr.plot_box_rmse(data, 'Fitness values, histogram per step (generation)',
                     len(ops),
                     optimiser.ngs,
                     'FitnessDevelopment', results_dir_boxplot)

    print("Plots have been created.")
    print("")
    print("")


####################################
# Plot Development of All Parameters
if cmdl_args.dump_plots:
    """
    Creates scatter plots of the development of each parameter over the 
    optimisation process. It reads the propti data 
    base file, based on information stored in the pickle file. 
    This functionality is focused on the usage of SPOTPY.
    """

    # TODO: Check for optimisation algorithm
    # TODO: Adjust output depending on optimisation algorithm

    print("")
    print("* Plot 'Likes' and Values.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir_scatter = check_directory([p1, p2, 'Scatter'])
    results_dir_boxplot = check_directory([p1, p2, 'Boxplot'])
    results_dir_para_gen = check_directory([p1, p2, 'Para_Gen'])
    results_dir_para_vs_fit = check_directory([p1, p2, 'Para_vs_Fitness'])

    # Extract data to be plotted.
    cols = ['like1', 'chain']
    pars = list()
    for parameter in ops:
        par_label = "par{}".format(parameter.place_holder)
        cols.append(par_label)
        pars.append(par_label)

    # Read data for the plots.
    data = pd.read_csv(db_file_name, usecols=cols)

    # Start plotting.
    # ---------------
    # Scatter plots of parameter development over the whole run.
    for c in cols[2:]:
        # Scatter plots of parameter development over the whole run.
        pr.plot_scatter(c, data, 'Parameter development: ', c,
                        results_dir_scatter, version=ver.ver_propti)

    # Scatter plot of fitness values.
    pr.plot_scatter('like1', data, 'Fitness value development',
                    'FitnessDevelopment', results_dir_scatter,
                    'Root Mean Square Error (RMSE)',
                    version=ver.ver_propti)

    # Plot values of best parameter set, by generation.
    pm.plot_best_para_generation(cols, data, len(ops), optimiser.ngs,
                                 results_dir_para_gen)

    # Box plot to visualise steps (generations).
    pr.plot_box_rmse(data, 'Fitness values, histogram per step (generation)',
                     len(ops),
                     optimiser.ngs,
                     'FitnessDevelopment', results_dir_boxplot)

    # Plot the parameter values against the fitness, colour coded by
    # repetition.
    pr.plot_para_vs_fitness(data_frame=data,
                            fitness_label=cols[0],
                            parameter_labels=pars,
                            file_path=results_dir_para_vs_fit,
                            version=ver.ver_propti)


    print("Plots have been created.")
    print("")
    print("")


if cmdl_args.calc_stat:
    """
    This functionality is very much work in progress.
    """
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


##########################################
# Plot Best Parameter Value, by Generation
if cmdl_args.plot_best_para_gen:

    """
    Plot the parameter values, for the best parameter set, of each generation.
    """

    print("")
    print("* Plot best values of a generation.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'Plots', 'Para_Gen'])

    # Collect the optimisation parameter names. Change format to match column
    # headers in propti_db, based on SPOTPY definition. Store headers in a list.
    cols = ['like1', 'chain']
    for p in ops:
        cols.append("par{}".format(p.place_holder))

    # Extract data to be plotted.
    data = pd.read_csv(db_file_name, usecols=cols)

    pm.plot_best_para_generation(cols, data, len(ops), optimiser.ngs,
                                 results_dir)

    print("")
    print("Plotting task completed.")
    print("")
    print("")


#########################
# Plot Fitness Semi-log x
if cmdl_args.plot_fit_semilogx:

    """
    Used to test functionality.
    """

    print("")
    print("* Fitness semi-log x.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir_semilogx_fitness = check_directory(['Analysis', 'Plots', 'Log'])

    # Collect the optimisation parameter names. Change format to match column
    # headers in propti_db, based on SPOTPY definition. Store headers in a list.
    cols = ['like1', 'chain']

    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plot of fitness values.
    pm.plot_semilogx_scatter('like1', data, 'Fitness value development',
                    'FitnessDevelopment', results_dir_semilogx_fitness,
                    'Root Mean Square Error (RMSE)')

    print("")
    print("Plot fitness semi-log x completed.")
    print("")
    print("")


################
# Data Extractor
if cmdl_args.extract_data:
    """
    Used to extract parameter sets, based on their fitness value.
    """

    print("")
    print("* Extract data.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir_best_para = check_directory([p1, p3, 'BestParameterGeneration'])

    # Collect the optimisation parameter names. Change format to match column
    # headers in propti_db, based on SPOTPY definition. Store headers in a list.
    cols = ['like1', 'chain']
    for p in ops:
        cols.append("par{}".format(p.place_holder))

    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plot of fitness values.
    pm.data_extractor(cols, data, len(ops), optimiser.ngs,
                      'BestParaExtraction', results_dir_best_para)

    print("")
    print("Extraction completed and file saved.")
    print("")
    print("")


##################################
# Create Input from Data Extractor
if cmdl_args.extractor_sim_input:
    """
    Takes the file that contains the results of the data extractor and builds 
    simulation input files from it. Files are stored in appropriate directory.
    """

    print("")
    print("* Creating input files from extracted data.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    extractor_dir = check_directory([p1, p3, 'ExtractorSimInput'])

    # Directory that shall contain the results from data_extractor.
    results_dir_best_para = os.path.join(cmdl_args.root_dir, p1, p3,
                                         'BestParameterGeneration')

    # Check if data collection exists.
    extr_file = os.path.join(results_dir_best_para, 'BestParaExtraction.csv')
    if os.path.isfile(extr_file):
        print("Data collection from data_extractor found.")
        print("")
    else:
        print("No data collection from data_extractor found.\n"
              "Please run the data_collector first.")
        print("")
        exit()

    # Collect simulation setup names.
    print("* Collect simulation setup names:")
    print("---")

    sim_setup_names = []
    for ssn in range(len(setups)):
        ssn_value = setups[ssn].name
        sim_setup_names.append(ssn_value)

        print("Setup {}: {}".format(ssn, ssn_value))
    print("---")
    print("")

    # Collect the optimisation parameter names. Change format to match column
    # headers in propti_db, based on SPOTPY definition. Store headers in a list.
    cols = []
    for p in ops:
        cols.append("par{}".format(p.place_holder))
    print(cols)
    # Collect parameter names
    print("* Collect parameter names and place holders:")
    print("---")

    para_names = []
    para_simsetup_complete = []
    para_name_list = []
    for s_i in range(len(setups)):

        # Place holder list
        para_ph_list = []

        # Collect model parameters, those which describe the simulation setup.
        # First, find all parameters and place holders.
        para_meta_simsetup = []
        for s_j in range(len(setups[s_i].model_parameter.parameters)):
            paras = setups[s_i].model_parameter.parameters

            para_name = paras[s_j].name
            para_name_list.append(para_name)

            para_ph = paras[s_j].place_holder
            para_ph_list.append(para_ph)

            # Compare the place holders with the optimisation parameters, to
            # determine if they are meta parameters.
            p_i = 'par{}'.format(para_ph)
            if p_i not in cols:
                # Store meta parameters (place holder and value) in list.
                para_meta_simsetup.append([para_ph, paras[s_j].value])

            print('Name: {}'.format(para_name))
            print('Place holder: {}'.format(para_ph))
        print("---")

        # Put meta lists into list which mirrors the simulation setups.
        para_simsetup_complete.append(para_meta_simsetup)
    print("")

    print("* Extract data from collection.")
    print("---")
    print("Read data collection file, please wait...")
    print("")

    # Read data collection from data_extractor.
    extr_data = pd.read_csv(extr_file, sep=',')

    #
    print("Number of data sets: {}".format(len(extr_data['repetition'])))
    for i in range(len(extr_data['repetition'])):

        print("* Fill templates")
        print("--------------")

        rep_value = int(extr_data.iloc[i]['repetition'])
        new_dir_rep = 'rep_{:08d}'.format(rep_value)
        check_directory([extractor_dir, new_dir_rep])
        print("Line: {}".format(i))
        print("Repetition value: {}".format(rep_value))
        print("")
        print("Parameters:")
        print("---")

        # Extract the parameter values of the best set. Store place holder and
        # parameter values in lists.
        opti_para = []
        for j in range(len(cols)):
            new_para_value = extr_data.at[i, cols[j]]
            print("{}: {}".format(para_name_list[j], new_para_value))
            opti_para.append([cols[j][3:], new_para_value])

        # Append optimisation parameter place holders and values to the
        # parameter lists, sorted by simulation setups.

        para_simsetup_complete_work = copy.deepcopy(para_simsetup_complete)
        for pssc in para_simsetup_complete_work:
            for para in opti_para:
                pssc.append(para)

        # Load templates from each simulation setup, fill in the values and
        # write the new input files in the appropriate directories.
        # Counter
        css = 0
        for simsetup in sim_setup_names:
            # Create new directories, based on simulation setup names.
            check_directory([extractor_dir, new_dir_rep, simsetup])

            # Load template.
            template_file_path = setups[css].model_template
            temp_raw = pbf.read_template(template_file_path)

            # Create new input files with best parameters,
            # based on simulation setups.
            for bestpara in para_simsetup_complete_work[css]:

                new_para_value = bestpara[1]

                if type(new_para_value) == float:
                    temp_raw = temp_raw.replace("#" + bestpara[0] + "#",
                                                "{:E}".format(new_para_value))
                else:
                    temp_raw = temp_raw.replace("#" + bestpara[0] + "#",
                                                str(new_para_value))

            # Write new input file with best parameters.
            bip = os.path.join(extractor_dir, new_dir_rep, simsetup,
                               simsetup + '_rep{}.fds'.format(
                                   int(extr_data.iloc[i]['repetition'])))
            pbf.write_input_file(temp_raw, bip)

            # Advance counter.
            css += 1

        para_simsetup_complete_work.clear()
        print("---")
        print("")

    print("Input files created.")
    print("")
    print("")


#############################
# Create Input for User Cases
if cmdl_args.create_case_input:

    """
    Templates of simulation input files are filled with data from the data 
    extractor. However, the templates are free to chose, thus means are provided
    to implement results from the IMP into different use cases.
    """

    print("")
    print("* Functionality testing.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    case_dir = check_directory(['Analysis', 'Cases'])

    # Directory that is supposed to contain the results from data_extractor.
    results_dir_best_para = os.path.join(cmdl_args.root_dir, 'Analysis',
                                         'BestParameter')

    # Check if data collection exists.
    extr_file = os.path.join(results_dir_best_para, 'BestParaExtraction.csv')
    if os.path.isfile(extr_file):
        print("Data collection from data_extractor found.")
        print("")
    else:
        print("No data collection from data_extractor found.\n"
              "Please run the data_collector first.")
        print("")
        exit()

    # Check if template exists.
    case_temp_name = 'C219_MT3_LargeDomain'
    template_file_path = '{}.fds'.format(case_temp_name)
    if os.path.isfile(template_file_path):
        print("Template for cases found.")
        print("")
    else:
        print("No template for cases found.\n"
              "Please provide a template.")
        print("")
        exit()

    # Read data collection from data_extractor.
    extr_data = pd.read_csv(extr_file, sep=',')

    #

    cols = list(extr_data)
    print(cols)

    print("Number of data sets: {}".format(len(extr_data['repetition'])))
    for i in range(len(extr_data['repetition'])):

        # Read case template.
        temp_raw = pbf.read_template(template_file_path)

        print("* Fill templates")
        print("--------------")

        rep_value = int(extr_data.iloc[i]['repetition'])
        new_dir_rep = 'rep_{:06d}'.format(rep_value)
        check_directory([case_dir, new_dir_rep])
        print("Line: {}".format(i))
        print("Repetition value: {}".format(rep_value))
        print("")
        print("Parameters:")
        print("---")

        for c in range(len(cols)):
            if "par" in cols[c]:
                if "insulator" in cols[c]:
                    # Divide insulator layer thickness by 2, for even split.
                    new_para_value = extr_data.at[i, cols[c]]/2
                else:
                    new_para_value = extr_data.at[i, cols[c]]

                print("{}: {}".format(cols[c], new_para_value))

                if type(new_para_value) == float:
                    temp_raw = temp_raw.replace("#" + cols[c][3:] + "#",
                                                "{:E}".format(new_para_value))
                else:
                    temp_raw = temp_raw.replace("#" + cols[c][3:] + "#",
                                                str(new_para_value))

        # Set character ID for file.
        rep_value = int(extr_data.iloc[i]['repetition'])
        temp_raw = temp_raw.replace("#chid#",
                                    "{}_rep{:06d}").format(case_temp_name,
                                                           rep_value)
        temp_raw = temp_raw.replace("#chid2#",
                                    "{}_rep{:06d}").format(case_temp_name,
                                                           rep_value)

        # Write new input file with best parameters.
        new_case_name = '{}_rep{:06d}.fds'.format(case_temp_name, rep_value)
        bip = os.path.join(case_dir, new_dir_rep, new_case_name)
        pbf.write_input_file(temp_raw, bip)
        print(len(temp_raw))

    print("")
    print("Functionality test completed.")
    print("")
    print("")


#######################
# Functionality testing
if cmdl_args.clean_db:

    """
    When using the restart functionality of SPOTPY, and having markers 
    written to the database this function helps to clean them up. 
    This function copies the original propti_db (`propti_db_original.csv`) to 
    the `Analysis/Databases/` directory, as a back-up.
    From this back-up it creates a reduced version where partly completed 
    generations, as well as the restart markers are removed.
    
    This new reduced version will then overwrite the `propti_db.csv` file.

    Focus is set on the SCEUA implementation of SPOTPY.
    """

    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    print("")
    print("* Cleaning database file '{}'.".format(db_file_name))
    print("----------------------")

    # Check if a directory for the result files exists. If not, create it.
    results_dir = check_directory([p1, 'Databases'])

    # Create backup of `propti_db.csv`, and append "_original" to its name.
    # Set file name of the back-up.
    db_original = os.path.join(results_dir, 'propti_db_original.csv')

    # Check if a back-up exists, create one if not.
    if os.path.isfile(db_original):
        print("* A back-up of the data base file already exists.")
        print("  Delete it manually if you want to create a new back-up.")
        print("")

    else:
        # Copy the original `propti_db.csv` file to a backup directory.
        shutil.copy2(db_file_name,
                     db_original)
        print("* New back-up data base file written.")
        print("")

        # Calculate generation size
        para_to_optimise = len(ops)
        num_complex = optimiser.ngs
        generation_size = int((2 * para_to_optimise + 1) * num_complex)
        print("Generation size: {}".format(generation_size))
        print("")

        # Count restart markers:
        restart_marker = "#Restart#"
        print("Restart marker: {}".format(restart_marker))
        print("")

        # Iterate over first column and look for restart_marker. Collect line
        # numbers with occurrences of restart_marker. Print line
        # number and content (the marker), to provide means for checking the
        # results.
        marker_occurrences = []
        print("Found markers:")
        print("line value")
        print("----------")
        marker_count = 0
        line_number = 0
        with open(db_original, 'r') as data_raw:
            for line in data_raw:
                if restart_marker in line:
                    print(line_number, line)
                    marker_occurrences.append(line_number)
                    marker_count += 1
                line_number += 1
        print("----------")
        print("Total markers: {}".format(marker_count))
        print("")

        # Provide an overview over the performed runs (restarts), the amount of
        # completed generations and the amount of individuals that do not fill
        # the last generation.
        gen_per_run = []
        print("Generations per run:")
        print("gen.\tres.\tleft")
        print("----------")
        # First run.
        gen = (marker_occurrences[0] - 1) // generation_size
        gen_per_run.append(gen)
        res = marker_occurrences[0] - 1 - gen * generation_size
        left = generation_size - res
        print("{}\t{}\t{}".format(gen, res, left))
        # Intermediate runs.
        for i in range(marker_count - 1):
            gen = (marker_occurrences[i + 1] - 1 - marker_occurrences[i]) \
                  // generation_size
            gen_per_run.append(gen)
            res = (marker_occurrences[i + 1] - 1 - marker_occurrences[i]) \
                  - gen * generation_size
            left = generation_size - res
            print("{}\t{}\t{}".format(gen, res, left))
        # Last run.
        lgi = (line_number - marker_occurrences[-1])
        gen = lgi // generation_size
        gen_per_run.append(gen)
        res = lgi - 1 - gen * generation_size
        left = generation_size - res
        print("{}\t{}\t{}".format(gen, res, left))
        print("----------")
        print("")

        ########
        # Get column labels.
        col_labels = list(pd.read_csv(db_original))
        print(col_labels)
        print(gen_per_run)

        # gen_per_run = [2, 1, 1, 0]
        #####

        #########
        # Note: This overwrites the original propti_db.csv, backup was created
        # at the beginning.
        db_reduced = db_file_name
        #########

        restart_count = 0
        indiv_count = 0
        print("* Processing the database file...")
        with open(db_original, 'r') as f:

            # Initialise the new database files.
            header = f.readline()
            #     with open(db_original, 'w') as dbc:
            #         dbc.write(header)
            with open(db_reduced, 'w') as dbr:
                dbr.write(header)

            # Iterate over every line.
            for line in f:
                # Check if restart marker is present, skip the line if true and
                # increase counter.
                if restart_marker not in line:
                    # Count the individuals to extract complete generations.
                    gen_count = gen_per_run[restart_count] * generation_size
                    if indiv_count < gen_count:
                        with open(db_reduced, 'a') as dbr:
                            dbr.write(line)
                        indiv_count += 1

                else:
                    print(restart_count, indiv_count, generation_size)
                    restart_count += 1
                    # Reset counter of individuals
                    indiv_count = 0

        print("")
        print("* Finished cleaning database file '{}'.".format(db_file_name))
        print("")
        print("")


#######################
# Plot Parameter Values (Sampling Range) against Fitness
if cmdl_args.plot_para_vs_fitness:

    """
    Plots the parameter values (sampling ranges) against the fitness values. 
    The scatter plot is colour coded by the repetition to allow to understand 
    at what part of the IMP run specific parameters and/or fitness values 
    were reached.
    """

    print("\n* Plot parameters vs. fitness.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'Plots', 'Para_vs_Fitness'])

    # Create lists of column headers, one to read the data and one for the
    # parameter plots.
    cols = ['like1']
    pars = list()
    for parameter in ops:
        par_label = "par{}".format(parameter.place_holder)
        cols.append(par_label)
        pars.append(par_label)

    # Read data for the plot.
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    pr.plot_para_vs_fitness(data_frame=data,
                            fitness_label=cols[0],
                            parameter_labels=pars,
                            file_path=results_dir,
                            version=ver.ver_propti)

    # Message to indicate that the job is done.
    print("--------------")
    print("Plot(s) have been created.\n\n")


#######################
# Functionality testing
if cmdl_args.func_test:

    """
    Scatter plot of fitness value (RMSE) development. It reads the PROPTI data 
    base file, based on information stored in the pickle file. 
    This functionality is focused on the usage of SPOTPY.
    """

    print("\n* Plot parameters vs. fitness.")
    print("----------------------")
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'Plots', 'Para_vs_Fitness'])

    # Create lists of column headers, one to read the data and one for the
    # parameter plots.
    cols = ['like1']
    pars = list()
    for parameter in ops:
        par_label = "par{}".format(parameter.place_holder)
        cols.append(par_label)
        pars.append(par_label)

    # Read data for the plot.
    data = pd.read_csv(db_file_name, usecols=cols)

    # Scatter plots of parameter development over the whole run.
    pr.plot_para_vs_fitness(data_frame=data,
                            fitness_label=cols[0],
                            parameter_labels=pars,
                            file_path=results_dir,
                            version=None)

    # Message to indicate that the job is done.
    print("--------------")
    print("Plot(s) have been created.\n\n")


    print("")
    print("* Functionality test completed.")
    print("")
    print("")


######################################
# Run Simulation of Best Parameter Set
if cmdl_args.create_new_database:
    """
    Extracts information from the `propti_db.csv` and creates a new database 
    used for subsequent analysis procedures. 
    """

    # Get file name of the `propti_db`
    db_file_name = os.path.join(cmdl_args.root_dir,
                                '{}.{}'.format(optimiser.db_name,
                                               optimiser.db_type))

    # Check if a directory for the result files exists. If not create it.
    results_dir = check_directory(['Analysis', 'Plots', 'Para_vs_Fitness'])

    # Create list of column headers to read the parameters.
    para_labels = list()
    for parameter in ops:
        para_label = "par{}".format(parameter.place_holder)
        para_labels.append(para_label)

    # New file name.
    new_db_file_name = "new_db"

    # Create new database file.
    ppm.create_base_analysis_db(db_file_name=db_file_name,
                                new_file_name=new_db_file_name,
                                output_dir=results_dir,
                                parameter_headers=para_labels,
                                fitness_headers=["like1"],
                                progress_headers=["chain"],
                                new_file_type="csv")


    print("")
    print("* Create new database file.")
    # print("----------------------")
    print("")
    print("")
