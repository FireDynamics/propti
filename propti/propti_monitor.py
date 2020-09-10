# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:39:13 2016

@author: thehnen; based on a script from belt
"""

import propti as pr
import numpy as np
import pandas as pd
import scipy.signal as sign
import matplotlib as mpl
from textwrap import wrap
import matplotlib.pyplot as plt

import re
import os

# mpl.use('pdf')


#%%
#
# # Set parameters for the plot which work when the plot is used in a
# # LaTeX document. It is used to provide correct font size and type used
# # in the document.
# mpl.rcParams['text.usetex']=True
# mpl.rcParams['font.size'] = 9
# mpl.rcParams['font.family'] = 'lmodern'
# mpl.rcParams['text.latex.unicode']=True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}',
#                                        r'\usepackage[T1]{fontenc}',
#                                        r'\usepackage{mathptmx}']


#%%


# def plot_best_para_generation(data_label, data_frame, plot_title, plot_text,
#                               para_to_optimise,
#                               num_complex,
#                               file_name=None, y_label=None, skip_lines=1):

def plot_best_para_generation(data_label, data_frame, para_to_optimise,
                              num_complex, file_path, file_type='png',
                              dpi_value=320, fontsize=13, scaling=0.88,
                              fig_size_x=6.5, fig_size_y=5.5):

    # Extract the total amount of individuals over all generations,
    # the very first individual will be skipped.
    individuals_total = len(data_frame['chain'].tolist()) - 1
    # Debugging:
    # print 'Individuals total:', individuals_total

    # Calculate generation size
    generation_size = int((2 * para_to_optimise + 1) * num_complex)
    print("Generation size: {}".format(generation_size))

    # Calculate number of full generations. If last generation is
    # only partly complete it will be skipped.
    generations = individuals_total // generation_size
    print("Generations: {}".format(generations))

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Prepare plotting of multiple plots in one diagram.
    plt.figure(figsize=(fig_size_x * scaling,
                        fig_size_y * scaling))

    # Find best fitness parameter per generation and collect them.
    local_best_locations = []
    for i in range(generations):
        start = 0 + i * generation_size
        end = 0 + (i+1) * generation_size

        local_best = data_frame.iloc[start:end]['like1'].idxmin()
        local_best_locations.append(local_best)
        print('Local best, gen. {}: {}'.format(i, local_best))
        print("Sample length: {}".format(
            data_frame.iloc[start:end]['like1'].size))

    for col_label in range(len(data_label[2:])):

        # Prepare plotting .
        # Collect data.
        x_values = []
        y_values = []

        for lbl in local_best_locations:
            x_values.append(lbl)
            y_values.append(data_frame.iloc[lbl][data_label[2 + col_label]])

        # Reset color cycle.
        # plt.gca().set_color_cycle(None)

        plt.plot(x_values, y_values, color='lightgrey')
        plt.plot(x_values, y_values, marker='.', linestyle='None', color='k')

        plt.title("Best value development per generation.")
        plt.xlabel("Individuals")
        plt.ylabel(data_label[2 + col_label])
        plt.grid()

        new_path = os.path.join(file_path,
                                data_label[2 + col_label] + '.' + file_type)
        print(new_path)
        plt.savefig(new_path, dpi=dpi_value)
        plt.show()
        plt.close()


def plot_scatter(data_label, data_frame, plot_title,
                 file_name=None, file_path=None, y_label=None, skip_lines=1,
                 version=None, file_type='png', dpi_value=320, fontsize=13,
                 scaling=0.88, fig_size_x=6.5, fig_size_y=5.5):

    """
    :param data_label: column label for Pandas data frame
    :param data_frame: name of the Pandas data frame
    :param plot_title: title of the plot
    :param version: conveys (Propti) version information to the plot
    :param file_name: name of the PDF-file to be created
    :param file_path: path to the location where the file shall be written
    :param y_label: label for the y-axis, default: data_label
    :param skip_lines: used to create plots while omitting the first lines
                       of the data frame
    :param file_type: type of the file to be saved
    :param dpi_value: dpi value for the image
    :param fontsize: font size for text in the plot
    :param scaling: scales the overall plot,
                    in concert with fig_size_x and fig_size_y
    :param fig_size_x: figure size along x-axis, in inch
    :param fig_size_y: figure size along y-axis, in inch

    :return: creates a plot and saves it as PDF-file
    """

    # Message to indicate that the plotting process has started.
    n_plots = 1
    print("")
    print("RMSE scatter plot(s):")
    print("Start plotting, {} task(s) in queue.".format(n_plots))
    print("--------------")

    # Extract the RMSE values from the data frame.
    # Skip first entry.
    rmse_values = data_frame[data_label][skip_lines:]

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Plot the RMSE values.
    fig = plt.figure(figsize=(fig_size_x * scaling,
                              fig_size_y * scaling))

    plt.plot(rmse_values, color='black', marker='.', linestyle='None')
    # Finish the plot
    # pl.rcParams['figure.figsize'] = 16, 12
    plt.xlabel('Individuals')

    if y_label is None:
        plt.ylabel(data_label)
    else:
        plt.ylabel(y_label)

    # Determine versions to provide them to the plot.
    if version is None:
        version_info = 'Not available.'
    else:
        version_info = version

    # Create plot title from file name.
    # print(type(repr(version_info)))
    plt.title(plot_title + file_name)
    # plt.figtext(0.6, 0.95, repr('Version: {}'.format(version_info)))
    plt.figtext(0.6, 0.95, 'Version: {}'.format(version_info))
    # plt.title(pr.Version().ver_propti + pr.Version().ver_fds, fontsize=6)
    plt.grid()

    # Check if a file name is provided, if it is a file will be
    # created.
    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path,
                                    file_name + '_scatter.' + file_type)
        else:
            new_path = os.path.join(file_name + '_scatter.' + file_type)
        plt.savefig(new_path, dpi=dpi_value)
    plt.close(fig)

    print("Plot '{}_scatter.pdf' has been created.".format(file_name))

    # Message to indicate that the job is done.
    print("--------------")
    print("Queue of plotting tasks finished.")
    print("")


def plot_scatter2(x_data, y_data, plot_title,
                  colour_data=None, file_name=None, file_path=None,
                  x_label=None, y_label=None, colour_label=None,
                  version=None, file_type='png', dpi_value=320, fontsize=13,
                 scaling=0.88, fig_size_x=6.5, fig_size_y=5.5):
    """

    :param x_data: the x-values for the plot
    :param y_data: the y-values of the plot
    :param plot_title: title of the plot
    :param colour_data: data used to colour the points
    :param file_name: name of the PNG-file to be created
    :param file_path: path to the location where the file shall be written
    :param x_label: label for the x-axis, default: data_label
    :param y_label: label for the y-axis, default: data_label
    :param colour_label: label of the data series that is used to colour the
        data points
    :param version: conveys (PROPTI) version information to the plot
    :param file_type: type of the file to be saved
    :param dpi_value: dpi value for the image
    :param fontsize: font size for text in the plot
    :param scaling: scales the overall plot,
                    in concert with fig_size_x and fig_size_y
    :param fig_size_x: figure size along x-axis, in inch
    :param fig_size_y: figure size along y-axis, in inch

    :return: creates a plot and saves it as PNG-file.
    """

    # Set a label for the color bar if none is provided.
    if colour_label is None:
        colour_label = 'None'

    # Message to indicate that the plotting process has started.
    print("\n* Scatter plot: {} vs. {}, coloured by {}.".format(y_label,
                                                                x_label,
                                                                colour_label))

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Create the scatter plots.
    fig = plt.figure(figsize=(fig_size_x * scaling,
                              fig_size_y * scaling))

    plt.scatter(x_data, y_data, c=colour_data)

    # Finish the plot
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Determine versions to provide them to the plot.
    if version is None:
        version_info = 'Not available.'
    else:
        version_info = version

    # Create plot title from file name.
    plt.title(plot_title)

    # Plot PROPTI version information.
    plt.figtext(0.6, 0.95, 'PROPTI Version: {}'.format(version_info))

    plt.grid()
    cbar = plt.colorbar()
    cbar.set_label(colour_label, labelpad=+1)
    # Check if a file name is provided, if it is a file will be
    # created.
    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path,
                                    file_name + '_scatter' + file_type)
        else:
            new_path = os.path.join(file_name + '_scatter' + file_type)
        plt.savefig(new_path, dpi=dpi_value)
    plt.close(fig)

    print("Plot '{}{}' has been created.".format(file_name,
                                                 '_scatter' + file_type))

    # # Message to indicate that the job is done.
    # print("--------------")
    # print("Done.\n")


def plot_para_vs_fitness(data_frame, fitness_label, parameter_labels,
                         file_path=None, version=None):
    """

    :param data_frame:
    :param fitness_label:
    :param parameter_labels:
    :param file_path:
    :param version:
    :return:
    """

    # Read fitness values.
    fitness_values = data_frame[fitness_label]

    # Scatter plots of parameter development over the whole run.
    for par in parameter_labels:
        file_name = par + '_vs_Fitness'
        plot_scatter2(x_data=fitness_values,
                      y_data=data_frame[par],
                      plot_title="{} vs. Fitness".format(par),
                      colour_data=range(len(fitness_values)),
                      file_name=file_name,
                      file_path=file_path,
                      x_label="Fitness values",
                      y_label=par,
                      colour_label="Repetition",
                      version=version)

    # # Message to indicate that the job is done.
    # print("--------------")
    # print("Done.\n")


def plot_semilogx_scatter(data_label, data_frame, plot_title,
                          file_name=None, file_path=None, y_label=None,
                          skip_lines=1, version=None, file_type='png',
                          dpi_value=320, fontsize=13, scaling=0.88,
                          fig_size_x=6.5, fig_size_y=5.5):

    """
    :param data_label: column label for Pandas data frame
    :param data_frame: name of the Pandas data frame
    :param plot_title: title of the plot
    :param version: conveys (Propti) version information to the plot
    :param file_name: name of the PDF-file to be created
    :param file_path: path to the location where the file shall be written
    :param y_label: label for the y-axis, default: data_label
    :param skip_lines: used to create plots while omitting the first lines
                       of the data frame
    :param file_type: type of the file to be saved
    :param dpi_value: dpi value for the image
    :param fontsize: font size for text in the plot
    :param scaling: scales the overall plot,
                    in concert with fig_size_x and fig_size_y
    :param fig_size_x: figure size along x-axis, in inch
    :param fig_size_y: figure size along y-axis, in inch

    :return: creates a plot and saves it as PDF-file
    """

    # Message to indicate that the plotting process has started.
    n_plots = 1
    print("")
    print("RMSE scatter plot(s):")
    print("Start plotting, {} task(s) in queue.".format(n_plots))
    print("--------------")

    # Extract the RMSE values from the data frame.
    # Skip first entry.
    rmse_values = data_frame[data_label][skip_lines:]

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Plot the RMSE values.
    fig = plt.figure(figsize=(fig_size_x * scaling,
                              fig_size_y * scaling))

    plt.semilogx(rmse_values, color='black', marker='.', linestyle='None')
    # Finish the plot
    # pl.rcParams['figure.figsize'] = 16, 12
    plt.xlabel('Individuals')

    if y_label is None:
        plt.ylabel(data_label)
    else:
        plt.ylabel(y_label)

    # Determine versions to provide them to the plot.
    if version is None:
        version_info = 'Not available.'
    else:
        version_info = version

    # Create plot title from file name.
    # print(type(repr(version_info)))
    plt.title(plot_title + file_name)
    # plt.figtext(0.6, 0.95, repr('Version: {}'.format(version_info)))
    plt.figtext(0.6, 0.95, 'Version: {}'.format(version_info))
    # plt.title(pr.Version().ver_propti + pr.Version().ver_fds, fontsize=6)
    plt.grid()

    # Check if a file name is provided, if it is a file will be
    # created.
    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path,
                                    file_name + '_semilogx_scatter' + file_type)
        else:
            new_path = os.path.join(file_name + '_semilogx_scatter' + file_type)
        plt.savefig(new_path, dpi=dpi_value)
    plt.close(fig)

    print("Plot '{}_semilogx_scatter.{}' has been created.".format(file_name,
                                                                   file_type))

    # Message to indicate that the job is done.
    print("--------------")
    print("Queue of plotting tasks finished.")
    print("")


def plot_semilogy_scatter(data_label, data_frame, plot_title,
                          file_name=None, file_path=None, y_label=None,
                          skip_lines=1, version=None, file_type='png',
                          dpi_value=320, fontsize=13, scaling=0.88,
                          fig_size_x=6.5, fig_size_y=5.5):

    """
    :param data_label: column label for Pandas data frame
    :param data_frame: name of the Pandas data frame
    :param plot_title: title of the plot
    :param version: conveys (Propti) version information to the plot
    :param file_name: name of the PDF-file to be created
    :param file_path: path to the location where the file shall be written
    :param y_label: label for the y-axis, default: data_label
    :param skip_lines: used to create plots while omitting the first lines
                       of the data frame
    :param file_type: type of the file to be saved
    :param dpi_value: dpi value for the image
    :param fontsize: font size for text in the plot
    :param scaling: scales the overall plot,
                    in concert with fig_size_x and fig_size_y
    :param fig_size_x: figure size along x-axis, in inch
    :param fig_size_y: figure size along y-axis, in inch

    :return: creates a plot and saves it as PDF-file
    """

    # Message to indicate that the plotting process has started.
    n_plots = 1
    print("")
    print("RMSE scatter plot(s):")
    print("Start plotting, {} task(s) in queue.".format(n_plots))
    print("--------------")

    # Extract the RMSE values from the data frame.
    # Skip first entry.
    rmse_values = data_frame[data_label][skip_lines:]

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Plot the RMSE values.
    fig = plt.figure(figsize=(fig_size_x * scaling,
                              fig_size_y * scaling))

    plt.semilogy(rmse_values, color='black', marker='.', linestyle='None')
    # Finish the plot
    # pl.rcParams['figure.figsize'] = 16, 12
    plt.xlabel('Individuals')

    if y_label is None:
        plt.ylabel(data_label)
    else:
        plt.ylabel(y_label)

    # Determine versions to provide them to the plot.
    if version is None:
        version_info = 'Not available.'
    else:
        version_info = version

    # Create plot title from file name.
    # print(type(repr(version_info)))
    plt.title(plot_title + file_name)
    # plt.figtext(0.6, 0.95, repr('Version: {}'.format(version_info)))
    plt.figtext(0.6, 0.95, 'Version: {}'.format(version_info))
    # plt.title(pr.Version().ver_propti + pr.Version().ver_fds, fontsize=6)
    plt.grid()

    # Check if a file name is provided, if it is a file will be
    # created.
    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path,
                                    file_name + '_semilogy_scatter.' + file_type)
        else:
            new_path = os.path.join(file_name + '_semilogy_scatter.' + file_type)
        plt.savefig(new_path, dpi=dpi_value)
    plt.close(fig)

    print("Plot '{}_semilogy_scatter.{}' has been created.".format(file_name,
                                                                   file_type))

    # Message to indicate that the job is done.
    print("--------------")
    print("Queue of plotting tasks finished.")
    print("")

# for multiple directories
#
# def plot_scatter_rmse(db_name, dir_names, plot_title,
#                       save_path, file_name = None):
#     # Message to indicate that the plotting process has started.
#     n_plots = len(dir_names)
#     print("")
#     print("RMSE scatter plot(s):")
#     print("Start plotting, {} task(s) in queue.".format(n_plots))
#     print("--------------")
#
#     # Traverse through the directories in DirNames.
#     for i in range(len(dir_names)):
#         # Change path of working directory to subdirectory (starting
#         # from the location this script lives in).
#         os.chdir(dir_names[i])
#
# #        # Get current working directory.
# #        cWD = os.getcwd()
# #        # Get file name of the shell script, which was used to start
# #        # the SLURM job. To be used as plot file name, later on.
# #        fileName = get_file_name(cWD,'.sh')
#
#         # Read in the results of the SCEUA process from CSV-file and
#         # convert into Pandas data frame.
#         # Set first row as header.
#         try:
#             sceua_data = pd.read_csv(db_name, header = 0)
#             # Debugging (skip first value):
#             #print SCEUAData['like1'][1:]
#         except:
#             message = "* File '{}', in directory '{}', not found."
#             print(message.format(db_name, dir_names[i]))
#             os.chdir('..')
#             continue
#
#         # Change the working directory back to the original location.
#         os.chdir('..')
#
#         # Extract the RMSE values from the data frame.
#         # Skip first entry.
#         rmse_values = sceua_data['like1'][1:]
#
#         # Plot the RMSE values.
#         fig = plt.figure()
#         plt.plot(rmse_values,color='black')
#
#         # Finish the plot
#         #pl.rcParams['figure.figsize'] = 16, 12
#         plt.xlabel('Individuals')
#         plt.ylabel('Root Mean Square Error (RMSE)')
#         # Create plot title from file name.
#         plt.title(plot_title[i] + '_' + file_name[i])
#         plt.grid()
#
#         # Check if a file name is provided, if it is a file will be
#         # created.
#         if not file_name[i] is None:
#             new_path = os.path.join(save_path,
#                                    'prefix_' + file_name[i] + '_scatter.pdf')
#
#             plt.savefig(new_path)
#         plt.close(fig)
#
#         print("Plot '{}_scatter.pdf' has been created.".format(file_name[i]))
#
#     # Message to indicate that the job is done.
#     print("--------------")
#     print("Queue of plotting tasks finished.")
#     print("")


def plot_box_rmse(df_name, plot_title, para_to_optimise,
                  num_complex, file_name=None, file_path=None,
                  file_type='png', dpi_value=320, fontsize=13,
                  scaling=0.88, fig_size_x=6.5, fig_size_y=5.5):

    """
    Create a collection of box plots, one for each generation of the sceua.
    Aimed to better visualise the fitness value development over the inverse
    modelling process (IMP). It is tailored to the SCEUA from the Spotpy package
    :param df_name: name of the Pandas data frame
    :param plot_title: title of the plot
    :param para_to_optimise: number of parameters of the IMP
    :param num_complex: number of complexes of the IMP, using Spotpy SCEUA
    :param file_name: name of the PDF-file to be created
    :param file_path: path to the location where the file shall be written
    :param file_type: type of the file to be saved
    :param dpi_value: dpi value for the image
    :param fontsize: font size for text in the plot
    :param scaling: scales the overall plot,
                    in concert with fig_size_x and fig_size_y
    :param fig_size_x: figure size along x-axis, in inch
    :param fig_size_y: figure size along y-axis, in inch

    :return: creates a plot and saves it as PDF-file
    """

    # Message to indicate that the plotting process has started.
    n_plots = 1
    print("")
    print("RMSE box plot(s):")
    print("Start plotting, {} task(s) in queue.".format(n_plots))
    print("--------------")

    # Extract the total amount of individuals over all generations,
    # the very first individual will be skipped.
    individuals_total = len(df_name['chain'].tolist())
    # Debugging:
    # print 'Individuals total:', individuals_total

    # Calculate generation size
    generation_size = int((2 * para_to_optimise + 1) * num_complex)
    print(generation_size)

    # Calculate number of full generations. If last generation is
    # only partly complete it will be skipped.
    generations = individuals_total // generation_size
    print(generations)

    # Debugging:
    # print 'Generations:', generations

    # Check if at least one full generation exists, if not print
    # a message and terminate this particular task and go on with
    # the remaining tasks. Otherwise proceed creating generation
    # plots.
    if generations < 1:
        print("* Individuals do not fill a whole generation for task:")
        print("* {}".format(file_name))
        print("* Task of plotting generation data stopped.")
        return
    # continue

    # Prepare list of lists to take the data for multiple boxplots, based
    # on the generations.
    data_series_multi = []

    # Slice the column with the RMSE data according to generations.
    for i in range(generations):
        # Calculate where to start slicing (skip very first value).
        slice_begin = generation_size * i + 1
        # Calculate where to stop slicing (skip very first value).
        slice_end = generation_size * (i + 1) + 1
        # Slice data and convert into a list.
        new_data = df_name['like1'][slice_begin:slice_end].tolist()
        # Collect the lists for later plotting.
        data_series_multi.append(new_data)

    # Set font size and font type for plot.
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams.update({'font.family': 'serif'})

    # Prepare plotting of multiple boxplots in one diagramm.
    multi_plot = plt.figure(figsize=(fig_size_x * scaling,
                                     fig_size_y * scaling))
    # Call the subplots.
    ax = multi_plot.add_subplot(111)
    # Create multiple boxplots
    ax.boxplot(data_series_multi)

    # Finish the plot
    # pl.rcParams['figure.figsize'] = 16, 12
    # Prepare list for x-tick labels.
    x_tick_labels = ['', ]
    # Create x-tick labels based on number of generations.
    for i in range(generations):
        x_tick_labels.append(str(i))
    # Set values for the x-ticks.
    plt.xticks(range(generations + 1),
               x_tick_labels, rotation='horizontal')
    plt.xlabel('Generations ({} individuals each)'.format(generation_size))
    plt.ylabel('Root Mean Square Error (RMSE)')
    # Create plot title from file name.
    plt.title(plot_title)
    plt.grid()

    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path,
                                    file_name + '_boxplot.' + file_type)
        else:
            new_path = os.path.join(file_name + '_boxplot.' + file_type)
        plt.savefig(new_path, dpi=dpi_value)
    plt.close(multi_plot)

    print("Plot '{}_boxplot.{}' has been created.".format(file_name,
                                                          file_type))
    # Message to indicate that the job is done.
    print("--------------")
    print("Queue of plotting tasks finished.")
    print("")


def data_extractor(data_label, data_frame, para_to_optimise, num_complex,
                   file_name=None, file_path=None, version=None,
                   best_data=True, get_limits=True):

    # Create copy of data label list to prevent conflicts.
    dl = data_label[:]

    # Extract the total amount of individuals over all generations,
    # the very first individual will be skipped.
    individuals_total = len(data_frame['chain'].tolist())
    # Debugging:
    # print 'Individuals total:', individuals_total

    # Calculate generation size
    generation_size = int((2 * para_to_optimise + 1) * num_complex)
    print("Generation size: {}".format(generation_size))

    # Calculate number of full generations. If last generation is
    # only partly complete it will be skipped.
    generations = individuals_total // generation_size
    print("Generations: {}".format(generations))

    #################

    # Collect data from the best parameter sets per generation.
    if best_data is True:
        print('Collect best parameter sets per generation.')

        # Find best fitness parameter per generation and collect them.
        # local_best_locations = []
        # print("sfdg", range(1))
        # for i in repetitions:
        #     print(i)
        #     start = 0 + i * generation_size
        #
        #     if generations < 1:
        #         end = individuals_total
        #         print("end", end)
        #     else:
        #         end = 0 + (i + 1) * generation_size
        #
        #     local_best = data_frame.iloc[start:end]['like1'].idxmax()
        #     local_best_locations.append(local_best)
        #     print('Local best, gen. {}: {}'.format(i, local_best))
        #     print("Sample length: {}".format(
        #         data_frame.iloc[start:end]['like1'].size))

        local_best_locations = []

        if generations < 1:
            start = 0
            end = individuals_total
            local_best = data_frame.iloc[start:end]['like1'].idxmin()
            local_best_locations.append(local_best)
            print('Local best, gen. {} not complete: {}'.format(0, local_best))
            print("Sample length: {}".format(
                data_frame.iloc[start:end]['like1'].size))
        else:
            for i in range(generations):
                print(i)
                start = 0 + i * generation_size
                end = 0 + (i + 1) * generation_size

                local_best = data_frame.iloc[start:end]['like1'].idxmin()
                local_best_locations.append(local_best)
                print('Local best, gen. {}: {}'.format(i, local_best))
                print("Sample length: {}".format(
                    data_frame.iloc[start:end]['like1'].size))

        # Collect corresponding data.
        new_data = []

        # For each row.
        for i in range(len(local_best_locations)):

            # For each column (parameter), create key-value pair and put
            # in dictionary.
            new_element = {}

            # Collect the number of the repetition to keep track where the
            # parameter sets are from.
            new_element.update({'repetition': local_best_locations[i]})

            for col_label in range(len(dl)):
                key = dl[col_label]
                value = data_frame.iloc[
                    local_best_locations[i]][dl[col_label]]
                new_element.update({key: value})

            # Collect dictionaries in list
            new_data.append(new_element)

    if best_data is False:
        print('Collect worst parameter sets per generation.')

        # Find worst fitness parameter per generation and collect them.
        local_worst_locations = []
        for i in range(generations):
            start = 0 + i * generation_size

            if generations < 1:
                end = individuals_total
            else:
                end = 0 + (i + 1) * generation_size

            local_worst = data_frame.iloc[start:end]['like1'].idxmin()
            local_worst_locations.append(local_worst)
            print('Local worst, gen. {}: {}'.format(i, local_worst))
            print("Sample length: {}".format(
                data_frame.iloc[start:end]['like1'].size))

        # Collect corresponding data.
        new_data = []

        # For each row.
        for i in range(len(local_worst_locations)):

            # For each column (parameter), create key-value pair and put
            # in dictionary.
            new_element2 = {}

            # Collect the number of the repetition to keep track where the
            # parameter sets are from.
            new_element2.update({'repetition': local_worst_locations[i]})

            for col_label in range(len(dl)):
                key = dl[col_label]
                value = data_frame.iloc[
                    local_worst_locations[i]][dl[col_label]]
                new_element2.update({key: value})

            # Collect dictionaries in list
            new_data.append(new_element2)

    # # Collect data from the best parameter sets per generation.
    # if best_data is True:
    #     # Find best fitness parameter per generation and collect them.
    #     local_best_locations = []
    #     for i in range(generations):
    #         start = 0 + i * generation_size
    #         end = 0 + (i + 1) * generation_size
    #
    #         local_best = data_frame.iloc[start:end]['like1'].idxmax()
    #         local_best_locations.append(local_best)
    #         print('Local best, gen. {}: {}'.format(i, local_best))
    #         print("Sample length: {}".format(
    #             data_frame.iloc[start:end]['like1'].size))
    #
    #     # Collect corresponding data.
    #     new_data = []
    #
    #     # For each row.
    #     for i in range(len(local_best_locations)):
    #
    #         # For each column (parameter), create key-value pair and put
    #         # in dictionary.
    #         new_element = {}
    #
    #         # Collect the number of the repetition to keep track where the
    #         # parameter sets are from.
    #         new_element.update({'repetition': local_best_locations[i]})
    #
    #         for col_label in range(len(data_label)):
    #             key = data_label[col_label]
    #             value = data_frame.iloc[
    #                 local_best_locations[i]][data_label[col_label]]
    #             new_element.update({key: value})
    #
    #         # Collect dictionaries in list
    #         new_data.append(new_element)

    #################

    # Construct pandas data frame from list of dicts, keep column labels .
    new_cols = dl.append('repetition')
    new_data_frame = pd.DataFrame(new_data, columns=new_cols)
    print(new_data_frame)

    # Write data frame to file.
    new_path = os.path.join(file_path, file_name + '.csv')
    new_data_frame.to_csv(new_path,
                          sep=',', encoding='utf-8')

# Multiple Directories
#
#
# def PlotBoxRMSE(dbName,DirNames,PlotTitle,GenerationSize,
#                 SavePath,fileName = None):
#     # Message to indicate that the plotting process has started.
#     nPlots = len(DirNames)
#     print ""
#     print "RMSE box plot(s):"
#     print "Start plotting, {} task(s) in queue.".format(nPlots)
#     print "--------------"
#
#     # Traverse through the directories in DirNames.
#     for i in range(len(DirNames)):
#         # Change path of working directory to subdirectory (starting
#         # from the location this script lives in).
#         os.chdir(DirNames[i])
#
#         # Read in the results of the SCEUA process from CSV-file and
#         # convert into Pandas data frame.
#         # Set first row as header.
#         try:
#             SCEUAData = pd.read_csv(dbName, header = 0)
#             # Debugging (Skip first value):
#             #print SCEUAData['like1'][1:]
#         except:
#             message = "* File '{}', in directory '{}', not found."
#             print message.format(dbName,DirNames[i])
#             os.chdir('..')
#             continue
#
#         # Change the working directory back to the original location.
#         os.chdir('..')
#         #Debugging:
#         #cWD2 = os.getcwd()
#         #print cWD2
#
#         # Extract the total amount of individuals over all generations,
#         # the very first individual will be skipped.
#         IndividualsTotal = len(SCEUAData['chain'].tolist()) - 1
#        # Debugging:
#         #print 'Individuals total:', IndividualsTotal
#
#         # Calculate number of full generations. If last generation is
#         # only partly complete it will be skipped.
#         Generations = IndividualsTotal / GenerationSize
#         # Debugging:
#         #print 'Generations:', Generations
#
#         # Check if at least one full generation exists, if not print
#         # a message and terminate this particular task and go on with
#         # the remaining tasks. Otherwise proceed creating generation
#         # plots.
#         if Generations < 1:
#             print "* Individuals do not fill a whole generation for task:"
#             print "* {}".format(fileName[i])
#             print "* Task of plotting generation data stopped."
#             continue
#
#         # Prepare list of lists to take the data for multiple boxplots, based
#         # on the generations.
#         dataSeriesMulti = []
#
#         # Slice the column with the RMSE data according to generations.
#         for i in range(Generations):
#             # Calculate where to start slicing (skip very first value).
#             sliceBegin = GenerationSize * i + 1
#             # Calculate where to stop slicing (skip very first value).
#             sliceEnd = GenerationSize * (i + 1) + 1
#             # Slice data and convert into a list.
#             newData = SCEUAData['like1'][sliceBegin:sliceEnd].tolist()
#             # Collect the lists for later plotting.
#             dataSeriesMulti.append(newData)
#
#         # Prepare plotting of multiple boxplots in one diagramm.
#         multiPlot = plt.figure()
#         # Call the subplots.
#         ax = multiPlot.add_subplot(111)
#         # Create multiple boxplots
#         ax.boxplot(dataSeriesMulti)
#
#         # Finish the plot
#         #pl.rcParams['figure.figsize'] = 16, 12
#         # Prepare list for x-tick labels.
#         xTickLabels = ['',]
#         # Create x-tick labels based on number of generations.
#         for i in range(Generations):
#             xTickLabels.append(str(i))
#         # Set values for the x-ticks.
#         plt.xticks(range(Generations + 1),
#                    xTickLabels, rotation='horizontal')
#         plt.xlabel('Generations')
#         plt.ylabel('Root Mean Square Error (RMSE)')
#         # Create plot title from file name.
#         plt.title(PlotTitle[i] + '_' + fileName[i])
#         plt.grid()
#
#         # Check if a file name is provided, if not a file will be created.
#         if not fileName[i]==None:
#
#             newPath = os.path.join(SavePath,
#                                    'prefix_'+ fileName[i] +'_boxplot.pdf')
#             # Debugging:
#             #print 'save path:'
#             #print newPath
#             # Save plot.
#             plt.savefig(newPath)
#         plt.close(multiPlot)
#
#         print "Plot '{}_boxplot.pdf' has been created.".format(fileName[i])
#     # Message to indicate that the job is done.
#     print "--------------"
#     print "Queue of plotting tasks finished."
#     print ""
