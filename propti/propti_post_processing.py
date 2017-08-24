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


# def