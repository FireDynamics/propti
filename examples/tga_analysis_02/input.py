# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser


# Goal of this file is to provide an example on how to run propti while
# taking multiple experiments into account. Thus, multiple simulation setups
# need to be created.

import numpy as np

# Import just for IDE convenience.
import propti as pr

# Set the character ID.
CHID = 'TGA_analysis_02'
TEND = 9360

# Define heating rates.
HeatingRatesTGA = [5, 10, 15]

# Define template file name.
template_file = "tga_analysis_02.fds"

# Define
experimental_data_file_list = ['tga_5K_exp.csv',
                               'tga_10K_exp.csv',
                               'tga_15K_exp.csv']

# Define the optimisation parameters.
# name: internal reference name
# place_holder: what to look for in the template, the '#' are added
#   automatically by propti
# min_value: defines the minimum limit of the range the optimisation is
#   allowed to sample from
# max_value: defines the maximum limit of the range the optimisation is
#   allowed to sample from
op1 = pr.Parameter(name='ref_temp_comp_01',
                   place_holder='rtc01',
                   min_value=200, max_value=400)
op2 = pr.Parameter(name='ref_rate_comp_01',
                   place_holder='rrc01',
                   min_value=0.001, max_value=0.01)
op3 = pr.Parameter(name='ref_temp_comp_02',
                   place_holder='rtc02',
                   min_value=300, max_value=600)
op4 = pr.Parameter(name='ref_rate_comp_02',
                   place_holder='rrc02',
                   min_value=0.001, max_value=0.01)

# Collect all the defined parameters from above, just for convenience.
set_of_parameters = [op1, op2, op3, op4]


# Definition of parameters, which is used by propti_prepare.py later on.
# It has no further meaning here.
ops = pr.ParameterSet(params=set_of_parameters)


# # define general model parameter, including optimisation parameter
# mps0 = pr.ParameterSet(params=set_of_parameters)
# mps0.append(pr.Parameter(name='heating rate',
#                          place_holder='hr',
#                          value=10))
#
# for i in range(len(HeatingRatesTGA)):
#     mps0.append(pr.Parameter(name='chid',
#                              place_holder='CHID',
#                              value='{}_{}K'.format(CHID, str(HeatingRatesTGA[
#                                                              i]))))


# Function to provide basic parameters for one simulation setup.
def create_mod_par_setup(para_set):
    # Provide optimisation parameters to the model parameter setups.
    ps = pr.ParameterSet(params=set_of_parameters)

    # Add different heating rates (5, 10, 15).
    ps.append(pr.Parameter(name='heating_rate_{}K'.format(str(HeatingRatesTGA[
                                                                  para_set])),
                           place_holder='hr',
                           value=HeatingRatesTGA[para_set]))

    # Add individual character ID to distinguish the simulation data.
    ps.append(pr.Parameter(name='chid',
                           place_holder='CHID',
                           value='{}_{}K'.format(CHID,
                                                 str(HeatingRatesTGA[
                                                         para_set]))))
    return ps

# Calls the above function to create multiple parameter sets for the different
# simulation setups. The parameter sets (objects) are then stored in a list.
model_parameter_setups = [create_mod_par_setup(i) for i in range(
    len(HeatingRatesTGA))]

model_parameter_setups.append(pr.ParameterSet(params=set_of_parameters))


# Create a list of relations between experimental and model (simulation) data,
# for each experimental data series. (Could also be nested, if there would be
# multiple repetitions for each experiment.)
r = []
for i in range(len(HeatingRatesTGA)):
    # Initialise a relation.
    relation = pr.Relation()
    # Information on simulation data.
    relation.model.file_name = "{}_{}K_tga.csv".format(CHID,
                                                       str(HeatingRatesTGA[i]))
    relation.model.label_x = 'Time'
    relation.model.label_y = 'MLR'
    relation.model.header_line = 1

    # The following parameters are the default values, which have no effect.
    relation.model.factor = 1  # Allows basic computation, could be used to
    # calculate heat release rate pr unit area.
    relation.model.offset = 0  # Add or substract number to shift the data.

    # Information on experimental data.
    relation.experiment.file_name = experimental_data_file_list[i]
    relation.experiment.label_x = 'Time'
    relation.experiment.label_y = 'MassLossRate'
    relation.experiment.header_line = 0

    # The following parameters are the default values, which have no effect.
    relation.experiment.factor = 1  # Allows basic computation, could be used to
    # calculate heat release rate pr unit area.
    relation.experiment.offset = 0  # Add or substract number to shift the data.

    # Define definition set for data comparison. Basically providing the
    # amount and position of data points in x-axis, by determining the range
    # (from 0. to TEND) and providing a delta between the points (12).
    relation.x_def = np.arange(0., TEND, 12)

    # Collect the different relations.
    r.append(relation)


# Initialise empty simulation setup sets.
setups = pr.SimulationSetupSet()

# Create simulation setups by joining all the necessary information:
# parameters, working directory, template file , relations and simulation
# software executable.
for i in range(len(HeatingRatesTGA)):
    s = pr.SimulationSetup(name='tga_analysis_02',
                           work_dir=
                           "{}_{}K".format(CHID, str(HeatingRatesTGA[
                                                                 i])),
                           model_template=template_file,
                           model_parameter=model_parameter_setups[i],
                           model_executable='fds653',
                           relations=r[i])

    setups.append(s)


print('** setups generated')


# Provide values for optimiser.
optimiser = pr.OptimiserProperties(algorithm='sceua',
                                   repetitions=150,
                                   ngs=len(set_of_parameters),
                                   # Sub-processes are used here for multiple
                                   # experimental conditions.
                                   num_subprocesses=len(HeatingRatesTGA),
                                   mpi=False)


print('** input file processed')
