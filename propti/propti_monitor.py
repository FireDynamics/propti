# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:39:13 2016

@author: thehnen; based on a script from belt
"""

import propti as pr
import numpy as np
import pandas as pd
import scipy.signal as sign
<<<<<<< HEAD
# import matplotlib as mpl

=======
import matplotlib as mpl
from textwrap import wrap
>>>>>>> master
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


<<<<<<< HEAD
def plot_scatter(data_label, data_frame, plot_title,
                 file_name=None, file_path=None, y_label=None, skip_lines=1):
=======
def plot_scatter(data_label, data_frame, plot_title, plot_text,
                 file_name=None, y_label=None, skip_lines=1):
>>>>>>> master

    """
    :param data_label: column label for Pandas data frame
    :param data_frame: name of the Pandas data frame
    :param plot_title: title of the plot
    :param file_name: name of the PDF-file to be created
    :param file_path: path to the location where the file shall be written
    :param y_label: label for the y-axis, default: data_label
    :param skip_lines: used to create plots while omitting the first lines
                       of the data frame
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

    # Plot the RMSE values.
    fig = plt.figure()
    plt.plot(rmse_values, color='black', marker='.', linestyle='None')
    # Finish the plot
    # pl.rcParams['figure.figsize'] = 16, 12
    plt.xlabel('Individuals')

    if y_label is None:
        plt.ylabel(data_label)
    else:
        plt.ylabel(y_label)
<<<<<<< HEAD

    plt.title(plot_title)

=======
    # Create plot title from file name.
    #print(type(repr(plot_text)))
    plt.title(plot_title + ' ' + file_name)
    plt.figtext(0.6, 0.95, repr(plot_text))
    # plt.title(pr.Version().ver_propti + pr.Version().ver_fds, fontsize=6)
>>>>>>> master
    plt.grid()

    # Check if a file name is provided, if it is a file will be
    # created.
    if file_name is not None:
        if file_path is not None:

            new_path = os.path.join(file_path, file_name + '_scatter.pdf')
        else:
            new_path = os.path.join(file_name + '_scatter.pdf')
        plt.savefig(new_path)
    plt.close(fig)

    print("Plot '{}_scatter.pdf' has been created.".format(file_name))

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
                  num_complex, file_name=None, file_path=None):

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
    individuals_total = len(df_name['chain'].tolist()) - 1
    # Debugging:
    #print 'Individuals total:', individuals_total

    # Calculate generation size
    generation_size = int((2 * para_to_optimise + 1) * num_complex)
    print(generation_size)

    # Calculate number of full generations. If last generation is
    # only partly complete it will be skipped.
    generations = individuals_total // generation_size
    print(generations)

    # Debugging:
    #print 'Generations:', generations

    # Check if at least one full generation exists, if not print
    # a message and terminate this particular task and go on with
    # the remaining tasks. Otherwise proceed creating generation
    # plots.
    if generations < 1:
        print("* Individuals do not fill a whole generation for task:")
        print("* {}".format(file_name))
        print("* Task of plotting generation data stopped.")
        return
    #continue

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

    # Prepare plotting of multiple boxplots in one diagramm.
    multi_plot = plt.figure()
    # Call the subplots.
    ax = multi_plot.add_subplot(111)
    # Create multiple boxplots
    ax.boxplot(data_series_multi)

    # Finish the plot
    #pl.rcParams['figure.figsize'] = 16, 12
    # Prepare list for x-tick labels.
    x_tick_labels = ['',]
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

            new_path = os.path.join(file_path, file_name + '_boxplot.pdf')
        else:
            new_path = os.path.join(file_name + '_boxplot.pdf')
        plt.savefig(new_path)
    plt.close(multi_plot)

    print("Plot '{}_boxplot.pdf' has been created.".format(file_name))
    # Message to indicate that the job is done.
    print("--------------")
    print("Queue of plotting tasks finished.")
    print("")


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
