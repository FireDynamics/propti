from .spotpy_wrapper import run_optimisation, create_input_file
from .data_structures import Parameter, ParameterSet, \
    SimulationSetupSet, SimulationSetup, Relation, DataSource, \
    OptimiserProperties

import logging
logging.basicConfig(filename='propti.log', level=logging.DEBUG)

pickle_prefix = 'propti.pickle'