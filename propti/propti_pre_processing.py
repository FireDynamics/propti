import re
import os
import sys
import shutil as sh
import logging

import propti as pr

import numpy as np
import pandas as pd
from scipy import stats
from scipy import interpolate
import scipy.signal as sign
from scipy.stats import norm
import matplotlib as mpl
mpl.use('pdf')


import matplotlib.pyplot as plt


setups = None  # type: pr.SimulationSetupSet
ops = None  # type: pr.ParameterSet
optimiser = None  # type: pr.OptimiserProperties


# This function takes a Pandas data frame and a list with header labels.
# Based on the header labels it looks for the shortest column. Afterwards
# it takes the smallest and largest values of the provided columns (per
# line) and collects them one list, each.
def calculate_min_mean_max_lists(data_frame, header_list):
    # Initialise the lists.
    list_min = []
    list_mean = []
    list_max = []

    # Determine the length of the shortest column.
    min_len_list = []

    # for column in range(len(header_list)):
    #
    #     a = len(data_frame[header_list[column]])
    #     min_len_list.append(a)

    for column in header_list:

        a = len(data_frame[column])
        min_len_list.append(a)

    min_len = min(min_len_list)

    # Iterate over all provided columns for the shortest length.
    for column in range(min_len):

        # Iterate over the columns by line and collect min and max values
        # in separate lists.
        interm_list = []
        for line in range(len(header_list)):
            interm_list.append(data_frame[header_list[line]][column])

        list_max.append(max(interm_list))
        list_mean.append(np.mean(interm_list))
        list_min.append(min(interm_list))

    return min_len, list_min, list_mean, list_max


def savgol_filter(x_values):
    filtered_data = sign.savgol_filter(x_values,
                                       37,
                                       3,
                                       deriv=0,
                                       delta=1.0,
                                       axis=-1,
                                       mode='interp',
                                       cval=0.0)
    return filtered_data


def interpolate_lists(lists, x_increment=1):

    n_lists = range(len(lists))

    min_collection = []
    for i in n_lists:
        min_collection.append(min(lists[i][0]))

    x_min = int(min(min_collection))

    x_new = np.arange(0, x_min, x_increment)


    for i in n_lists:
        f = interpolate.interp1d(lists[i][0], lists[i][1])
        ynew = f(x_new)
        pass




    xnew = np.arange(0, 5.5, 0.1)
    ynew = f(xnew)

































