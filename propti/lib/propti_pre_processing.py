import re
import os
import sys
import shutil as sh
import logging

from .. import lib as pr

import statistics as stat
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


def interpolate_lists(raw_lists, x_increment=1, window=21, poly_order=3,
                      new_data_file='proc_data',
                      plot_file_name='average_smooth_plot',
                      plot_title='Averaged and Sav-Gol smoothed',
                      x_label='x label',
                      y_label='y label'):

    n_lists = range(len(raw_lists))

    x_max_collection = []
    for i in n_lists:
        x_max_collection.append(max(raw_lists[i][0]))

    # Determine the length of the shortest data series to fit the other to it.
    x_min = int(min(x_max_collection))
    print('max col: {}'.format(x_max_collection))
    print('x_min: {}'.format(int(x_min)))
    x_new = np.arange(0, x_min, x_increment)

    # Interpolate each data series to fit to the same x-values.
    interpolated_data = [x_new]
    for i in n_lists:
        f = interpolate.interp1d(raw_lists[i][0], raw_lists[i][1])
        y_new = f(x_new)
        interpolated_data.append(y_new)

    # Calculate the average over all lists per x-value.
    data_mean = []
    data_median = []
    for i in range(len(interpolated_data[0])):
        data_to_be_averaged = []
        for j in n_lists[0:]:
            new_element = interpolated_data[j+1][i]
            data_to_be_averaged.append(new_element)

        element_mean = stat.mean(data_to_be_averaged)
        element_median = stat.median(data_to_be_averaged)

        data_mean.append(element_mean)
        data_median.append(element_median)

    # Smoothing of the new data, using Savitzky-Golay filter.
    data_smoothed = sign.savgol_filter(data_mean,
                                       window,
                                       poly_order)

    d1 = sign.savgol_filter(data_median,
                            window,
                            poly_order)
    processed_data1 = [x_new, d1]

    processed_data = [x_new, data_smoothed]

    # Create Pandas DataFrame with the new values and save them as CSV.
    proc1 = np.vstack((x_new, data_smoothed))
    proc2 = pd.DataFrame.from_records(proc1.transpose(),
                         columns=['newx', 'newy']).set_index('newx')
    proc2.to_csv('{}.csv'.format(new_data_file))
    print(proc2.head())

    fig = plt.figure()
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i in n_lists:
        plt.plot(raw_lists[i][0], raw_lists[i][1],
                 color='gray', label='Raw')

    plt.plot(processed_data[0], processed_data[1],
             color='black', label='Processed mean')

    plt.plot(processed_data1[0], processed_data1[1],
             color='red', label='Processed median', linestyle='--')

    plt.grid()
    plt.legend(loc='best')
    plt.savefig(plot_file_name)
    plt.close(fig)

    return interpolated_data



































