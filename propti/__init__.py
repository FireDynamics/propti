from .spotpy_wrapper import run_optimisation, create_input_file
from .data_structures import Parameter, ParameterSet, \
    SimulationSetupSet, SimulationSetup, Relation, DataSource, \
    OptimiserProperties

import logging

#########
# LOGGING
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='propti.log',
                    filemode='w')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

# tell the handler to use this format
console.setFormatter(formatter)

# add the handler to the root logger
logging.getLogger('').addHandler(console)

###########
# CONSTANTS

# TODO: respect this variable in scripts
pickle_prefix = 'propti.pickle'