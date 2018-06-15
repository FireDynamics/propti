# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser

"""
Goal of this file is to provide an example on how to get started with PROPTI.
Here a basic, simple setup is provided.

The baseline concept:
---------------------
In general, it is necessary to define the "optimisation parameters". These
are the parameters, which are changed by the optimisation algorithm during
the inverse modelling process (IMP).
Files, like the template for the simulation software and the experimental
data, need to be provided.
Relationships between simulation results and experimental data need to be
defined.
All this information needs to be bundled, together with the desired
simulation software, into a simulation setup.
Finally, the desired optimisation algorithm needs to be chosen, and provided
with the necessary parameters.
"""

import numpy as np

# Import just for IDE convenience.
import propti as pr


"""
At first we provide some general information that will be needed throughout 
this input file. These are items like template file names or target data files.
"""
# Input file template for the simulation software.
template_file = "template_tga_01.fds"

# File containing the experimental data that is used as target during the
# inverse modelling process.
exp_data_file = "tga_experimental_data.csv"

# Character ID used to identify the files connected to one simulation.
CHID = 'TGA_analysis_01'

#
TEND = 9360

"""
For the optimisation process parameters need to be defined that are to be 
changed, or worked on, by the optimisation algorithm. These parameters have 
been identified by the user, previously of performing the IMP, to influence 
the behaviour one is interested in.
They get a name, a place holder, as well as limits to describe a range of 
values the optimisation algorithm is allowed to sample from during the IMP.

name: Internal, human-readable, reference name
place_holder: What to look for in the template, the '#' characters  are added
  automatically by PROPTI
min_value: Defines the minimum limit of the range the optimisation algorithm is
  allowed to sample from
max_value: Defines the maximum limit of the range the optimisation algorithm is
  allowed to sample from
"""
# Define the optimisation parameters.
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


# Definition of parameters, which is used by `propti_prepare.py` later on.
# It has no further meaning here.
ops = pr.ParameterSet(name="Optimisation parameters",
                      params=set_of_parameters)


"""
Besides the optimisation parameters, also model parameters need to be 
provided. These model parameters can describe things like the environmental 
conditions appropriate to the simulation, here it is the heating rate. They can 
furthermore provide meta information, like a character ID to label the 
simulation or things like simulation end times or data dump intervals. 
"""
# Definition of general model parameters, including optimisation parameters.
mps0 = pr.ParameterSet(name="Environment and\n"
                            "Optimisation parameters",
                       params=set_of_parameters)
mps0.append(pr.Parameter(name='heating rate', place_holder='hr', value=10))
mps0.append(pr.Parameter(name='chid', place_holder='CHID', value=CHID))


"""
The optimisation algorithm needs information on what its target is, and what 
data from the simulation (model) it needs to be compared to. This connection 
is defined using relations. The relation is divided into two groups, 
for the model (simulation) and the experiment. However, the underlying 
parameters are the same. 
It is expected that the data files for both are comma separated value files 
(csv). Labels need to be defined, that describe the data series (x/y values), 
as well as a header line where to find the labels.
It is further possible to provide a definition range. This is 
specifically interesting, when model and simulation data were taken with 
different time intervals between their respective data points.  
"""
# Define model-experiment data relation
r = pr.Relation()
r.model.file_name = "{}_tga.csv".format(CHID)
r.model.label_x = 'Time'
r.model.label_y = 'MLR'
r.model.header_line = 1
r.experiment.file_name = exp_data_file
r.experiment.label_x = 'Time'
r.experiment.label_y = 'MassLossRate'
r.experiment.header_line = 0

# Define definition range for data comparison
r.x_def = np.arange(0., TEND, 12)


"""
The term "simulation setup" is used for a complete collection of parameters 
that describe a desired simulation as a whole. It could be compared as the 
set of parameters that would describe an experiment. 
This collection then contains information of the parameters, input file 
templates, simulation software, experimental data, the model-experiment data 
relations and the working directories.
In this specific case the Fire Dynamics Simulator FDS was chosen as 
simulation software.   
"""
# Create simulation setup object
s = pr.SimulationSetup(name='tga_analysis_01',
                       work_dir='tga_analysis_run_01',
                       model_template=template_file,
                       model_parameter=mps0,
                       model_executable='fds',
                       relations=r)


"""
It is possible to create multiple simulation setups if the set of 
optimisation parameters is desired to be tested under different conditions. 
However, this functionality will be discussed in a further example.

Still, it is necessary to store the simulation setup, created above, into a 
simulation setup set. 
"""
# Define empty simulation setup set
setups = pr.SimulationSetupSet()
# Append above object to simulation setup set
setups.append(s)


"""
Finally, some information needs to be provided to the optimiser. It would be 
sufficient to define an algorithm and the desired amount repetitions for a 
basic IMP run. As example the shuffled complex evolutionary algorithm, 
implemented in SPOTPY was chosen. 

Below, some of the default values are highlighted to show some of the options 
provided to the user. 
Specifically, two may be important the `backup_every` and `db_precision`.
backup_every: A number of repetitions conducted, after which a back-up file 
    is written (by SPOTPY, in this case). This file provides means to restart 
    IMP runs, in an event of a system crash, for instance. Keep in mind, 
    how it works in detail, depends on the chosen algorithm. For SCEUA, 
    it means, if the this number of repetition is reached, the break point 
    will be written when the recent generation is finished. 
db_precision: This allows the user to control the precision of the values 
    that are written to the data base.
The data base (db) file can be adjusted as well, by `db_name` and `db_type`. 
It is recommended to leave the type as a comma separated value (csv) file, 
if one is not sure what to chose. 
"""
# Provide parameter values for the optimiser, some of the default values are
# highlighted here.
optimiser = pr.OptimiserProperties(algorithm='sceua',
                                   repetitions=1000,
                                   backup_every=100,
                                   db_name='propti_db',
                                   db_type='csv',
                                   db_precision=np.float64,
                                   mpi=False)


print('** input file processed')
