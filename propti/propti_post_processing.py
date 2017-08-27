# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:39:13 2016

@author: thehnen; based on a script from belt
"""

import re
import os
import sys
import shutil as sh
import logging

import propti as pr

import numpy as np
import pandas as pd
import scipy.signal as sign
from scipy.stats import norm
from scipy import stats
import matplotlib as mpl
mpl.use('pdf')


import matplotlib.pyplot as plt


setups = None  # type: pr.SimulationSetupSet
ops = None  # type: pr.ParameterSet
optimiser = None  # type: pr.OptimiserProperties


def run_best_para(setups_bp, ops_bp, optimiser_bp, pickle_object):
    print(setups_bp, ops_bp, optimiser_bp)

    for s in setups_bp:

        input_file_directory = s.work_dir

        root_dir = os.path.dirname(os.path.abspath(pickle_object))
        cdir = os.path.join(root_dir, s.best_dir)

        # create best parameter simulation directories
        if not os.path.exists(cdir):
            os.mkdir(cdir)

        # copy model template
        pt = os.path.abspath(input_file_directory)
        sh.copy(os.path.join(pt, s.model_template.split('/')[-1]), cdir)

        s.model_template = os.path.join(cdir,
                                        os.path.basename(s.model_template))

        # copy all experimental data
        for r in s.relations:
            sh.copy(os.path.join(pt, r.experiment.file_name.split('/')[-1]),
                    cdir)
            r.experiment.file_name = \
                os.path.join(cdir,
                             os.path.basename(
                                 r.experiment.file_name.split('/')[-1]))

    # check for potential non-unique model input files
    in_file_list = []
    for s in setups_bp:
        tpath = os.path.join(s.work_dir, s.model_input_file)
        logging.debug("check if {} is in {}".format(tpath, in_file_list))
        if tpath in in_file_list:
            logging.error("non unique module input file path: {}".format(tpath))
            sys.exit()
        in_file_list.append(tpath)

    for s in setups_bp:
        pr.create_input_file(s, work_dir='best')

    pr.run_simulations(setups_bp, best_para_run=True)
    pass


def plot_template(exp_data, sim_data, legend_labels=None,
                  plot_labels=None, pdf_name='Plot name', n_colors=10):

    if plot_labels is None:
        print('* Specify plot_labels=[x-label, y-label, title], all as string.')
        plot_labels = ['x-label', 'y-label', 'title']

    if legend_labels is None:
        print('* Specify legend_labels as list of strings.')
        legend_labels = ['dummy label']

    # Prepare plotting of multiple plots in one diagram.
    multi_plot = plt.figure()

    # Call the subplots.
    ax = multi_plot.add_subplot(111)

    # Set default color map to viridis.
    # https://www.youtube.com/watch?v=xAoljeRJ3lU&feature=youtu.be
    colormap = plt.get_cmap('viridis')
    ax.set_color_cycle([colormap(k) for k in np.linspace(0, 1, n_colors)])
    # ax.set_prop_cycle('viridis', plt.cm.spectral(np.linspace(0, 1, 30)))

    for i in range(len(exp_data)):
        # Create multiple plots
        ax.plot(exp_data[i][0],
                exp_data[i][1],
                linestyle='-.',
                color=colormap(i))

        ax.plot(sim_data[i][0],
                sim_data[i][1],
                linestyle='-',
                color=colormap(i))

    ax.legend(legend_labels)

    plt.xlabel(plot_labels[0])
    plt.ylabel(plot_labels[1])
    # Create plot title from file name.
    plt.title(plot_labels[2])
    plt.grid()

    plt.savefig(pdf_name + '.pdf')
    plt.close(multi_plot)
    print('Plot saved.')
    print('')
    pass


def plot_hist(data_label, data_frame, file_name, bin_num=100, y_label=None):

    """

    :param data_label: label of the parameter (column label for pandas)
    :param data_frame: pandas data frame in which to look for the column
    :param file_name: name which will be given to the PDF-file
    :param bin_num: number of bins for the histogram, default: 100
    :param y_label: label for the y-axis, default: data_label
    :return: saves histogram plot as PDF-file
    """

    # Prepare data for plot.
    x = data_frame[data_label]

    # Plot histogram of data points.
    plt.hist(x, bins=bin_num)

    plt.xlabel('Individuals')
    if y_label is None:
        plt.ylabel(data_label)
    else:
        plt.ylabel(y_label)

    if file_name is not None:
        target_path = os.path.join(file_name + '_' + data_label + '.pdf')
        plt.savefig(target_path)
        plt.close()


# TODO: decouple the calculation of descriptive statistics
# TODO: create function to call specific calculation methods
'''
def descriptive_statistics(complete_sample, offset, data_label, n=1,
                           skip_zero=False):

    # work in progress

    # Determine how much a given parameter changes during an optimisation
    # run. For small changes set parameter to a fixed value, which is the
    # mean of the sample (descriptive statistics) without the first n
    # generations.
    #
    # It calculates and returns a couple of descriptive statistics:
    # mean, standard deviation, skewness, mode,
    #
    # CompleteSample is supposed to be an array-like (Pandas data frame
    # column).
    # Offset is supposed to be an integer which will exclude the first n
    # individuals from the analysis. It is meant to be a number which
    # represents the first n generations, to exclude strong fluctuations
    # during the beginning of the optimisation process.
    #

    # Prepare data for plot.
    sub_set = complete_sample[offset:].tolist()

    # Cut away the burn-in (generation 0).
    if skip_zero is True:
        complete_sample = complete_sample[GenerationSize:].tolist()

    # Some descriptive statistics on the data set, for the whole data
    # set and the subset:
    #
    # Prepare list for statistic data.
    statistic_data = []

    # Calculate mean (mu) and standard deviation (std) of both, the
    # complete sample, as well as the subset.
    mu_complete, std_complete = norm.fit(complete_sample)
    mu_sub_set, std_sub_set = norm.fit(sub_set)
    # fit = norm.pdf(sub_set, mu_sub_set, std_sub_set)
    statistic_data.append(mu_complete)
    statistic_data.append(std_complete)
    statistic_data.append(mu_sub_set)
    statistic_data.append(std_sub_set)

    # Calculate skewness.
    skew_complete = stats.skew(complete_sample, axis=0, bias=True)
    skew_sub_set = stats.skew(sub_set, axis=0, bias=True)
    statistic_data.append(skew_complete)
    statistic_data.append(skew_sub_set)

    # Calculate kurtosis.
    kurt_complete = stats.kurtosis(complete_sample, axis=0, fisher=True,
                                   bias=True)
    kurt_sub_set = stats.kurtosis(sub_set, axis=0, fisher=True, bias=True)
    statistic_data.append(kurt_complete)
    statistic_data.append(kurt_sub_set)

    # Calculate mode.
    mode_complete = stats.mode(complete_sample, axis=0)
    mode_sub_set = stats.mode(sub_set, axis=0)
    statistic_data.append(mode_complete)
    statistic_data.append(mode_sub_set)

    print("Parameter: ", data_label)

    # Calculate range of n standard deviations around the mean value of
    # the subset.
    high = mu_sub_set + n * std_sub_set
    low = mu_sub_set - n * std_sub_set
    print("High subset: ", high)
    print("Mean complete: ", mu_complete)
    print("Low subset: ", low)
    print("Mean complete: ", statistic_data[0])
    print("Mean subset: ", statistic_data[1])
    print("Std deviation complete: ", statistic_data[2])
    print("Std deviation subset: ", statistic_data[3])
    print("Skewness complete: ", statistic_data[4])
    print("Skewness subset: ", statistic_data[5])
    print("Kurtosis complete: ", statistic_data[6])
    print("Kurtosis subset: ", statistic_data[7])
    print("Mode complete: ", statistic_data[8][0])
    print("Mode subset: ", statistic_data[9][0])

    return reduced, rejected, rejectedParaName, \
           rejectedParaValue, reducedParaName, reducedParaValue, \
           statistic_data


def calc_mode():
    # Calculate mode.
    mode_complete = stats.mode(complete_sample, axis=0)
    mode_sub_set = stats.mode(sub_set, axis=0)
    statistic_data.append(mode_complete)
    statistic_data.append(mode_sub_set)
    pass

'''


def calc_pearson_coefficient(data_series):
    corr_mat = np.corrcoef(data_series)
    return corr_mat


def collect_best_para_multi(data_file, label_list, distance=0.5e-4):
    """

    :param data_file: Assumed to be a CSV-file and containing the data base
        information provided by SPOTPY
    :param label_list: List of labels (string) which are indexing the columns
        of the shape:
        [fitness values, parameter_1, parameter_2, ..., parameter_n]
    :param distance: Half of the range in which to look for parameter sets
        around the best fitness value
    :return: para_collection: Pandas DataFrame with the collected parameter
        sets and their respective fitness values
    """

    # Read Pandas DataFrame and convert the content of one column into
    #  a numpy array.
    fit_vals_raw = pd.read_csv(data_file, usecols=label_list)
    fit_vals = fit_vals_raw[label_list[0]].values

    # Find max value in the array.
    fit_max = max(fit_vals)

    # Calculate the range in which to collect the samples.
    upper = fit_max + distance
    lower = fit_max - distance
    print(upper)
    print(lower)

    print('Start comparison:')

    # Collect indices and values.
    multi_fit = []
    row_indices = []
    for num_i in range(len(fit_vals)):
        new_element = []
        print(fit_vals[num_i])
        if lower <= fit_vals[num_i] <= upper:
            new_element.append(num_i)
            row_indices.append(num_i)
            new_element.append(fit_vals[num_i])
            print(fit_vals[num_i])
            multi_fit.append(new_element)
        else:
            print("False")

    print('')
    print('-------------')
    print('Range around the best fitness value')
    print('Best fitness: {}'.format(fit_max))
    print('Distance: {}'.format(distance))
    print('Upper bound: {}'.format(upper))
    print('Lower bound: {}'.format(lower))
    print('')
    for i in range(len(multi_fit)):
        print('    ', fit_vals_raw.loc[multi_fit[i][0], 'like1'])
        print(multi_fit[i])
        print('')
    print('-------------')

    # Create a Pandas DataFrame with the samples which are
    # within the range.
    para_collection = fit_vals_raw.loc[row_indices, label_list]

    return para_collection


def plot_best_sim_exp(setup_plot, pickle_object):

    root_dir = os.path.dirname(os.path.abspath(pickle_object))
    cdir = os.path.join(root_dir, setup_plot.best_dir)

    # Check if best parameter simulation directories exist
    if not os.path.exists(cdir):
        print('* No directory of best parameter simulation found.')
        print('* Hint: Use run_best_para method for that simulation.')
        return

    # Show relation information.
    for r in setup_plot.relations:
        print(r)

    # Determine amount of relations to give every plot its own color
    # without duplicates.
    lr = len(setup_plot.relations)

    # Extract data from simulation and experiment to be plotted.
    model_data = []
    experimental_data = []
    for r in setup_plot.relations:

        mod_file = os.path.join(cdir, r.model.file_name)
        model_data_raw = pd.read_csv(mod_file,
                                     header=r.model.header_line,
                                     usecols=[r.model.label_x,
                                              r.model.label_y])

        experimental_data_raw = pd.read_csv(r.experiment.file_name,
                                            header=r.experiment.header_line,
                                            usecols=[r.experiment.label_x,
                                                     r.experiment.label_y])

        md_interm = [model_data_raw[r.model.label_x].tolist(),
                     model_data_raw[r.model.label_y].tolist()]

        ed_interm = [experimental_data_raw[r.experiment.label_x].tolist(),
                     experimental_data_raw[r.experiment.label_y].tolist()]

        model_data.append(md_interm)
        experimental_data.append(ed_interm)

    leg_lab = ['experiment', 'simulation']
    plot_template(experimental_data, model_data,legend_labels=leg_lab,
                  n_colors=lr)

























