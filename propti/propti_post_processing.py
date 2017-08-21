# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:39:13 2016

@author: thehnen; based on a script from belt
"""

import numpy as np

import pandas as pd
import scipy.signal as sign
from scipy.stats import norm
from scipy import stats
import matplotlib as mpl
mpl.use('pdf')


import matplotlib.pyplot as plt

import re
import os


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