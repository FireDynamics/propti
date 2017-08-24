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

