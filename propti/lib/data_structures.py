import os
import sys
import logging
import copy
import ast
import numpy as np
import pandas as pd
import subprocess
from typing import Union
import spotpy
from .fitness_methods import FitnessMethodInterface
from typing import Union

from .. import lib as pr

from typing import List

# Reads the script's location. Used to access the propti version number from the
# git repo.
script_location = os.path.dirname(os.path.realpath(__file__))


#################
# OPTIMISER CLASS
class OptimiserProperties:
    """
    Stores optimiser parameter. They describe parameters that are used by the
    optimiser, e.g. SPOTPY.
    """

    def __init__(self,
                 algorithm: str = 'sceua',
                 repetitions: int = 1,
                 backup_every: int = 100,
                 max_loop_inc: int = None,
                 ngs: int = None,  # will be set to len(ops) during
                 # propti_prepare, if no value is provided
                 eb: int = None,
                 db_name: str = 'propti_db',
                 db_type: str = 'csv',
                 db_precision=np.float64,
                 num_subprocesses: int = 1,
                 mpi: bool = False):
        """
        Constructor.
        :param algorithm: choose spotpy algorithm, default: sceua,
            range: [sceua]
        :param repetitions: number of sampling repetitions, default: 1
        :param backup_every: How many repetitions before backup is performed,
            default: 100
        :param max_loop_inc: How many loop iteration per sample call, default: None
        :param ngs: number of complexes, if None then set to len(para),
            only applys to SCEUA
            default: None
        :param ngs: individuals/2 for each generation, if None then set to 48,
            only applys to ABC and FSCABC
            default: None
        :param db_name: name of spotpy database file, default: propti_db
        :param db_type: type of database, default: csv, range: [csv]
        :param db_precision: desired precision of the values to be written into
            the data base, default: np.float64
        :param num_subprocesses: Used to set the number of sub processes
            that need to be run for each task.
            For example, when three experiments with different
            conditions have been performed, for the same material, each one
            would require one subprocess -> num_subprocesses = 3.
            Default: 1
        :param mpi: Parameter that gets provided to SPOTPY,
            via spotpy_wrapper.py (run_optimisation), to the _algorithm
            parameter 'parallel'. If set to True 'parallel' is set to 'mpi'
            otherwise it is set to 'seq' (use one core).
            Default: False
        """
        self.algorithm = algorithm
        self.repetitions = repetitions
        self.backup_every = backup_every
        self.max_loop_inc = max_loop_inc
        self.ngs = ngs
        self.eb = eb
        self.db_name = db_name
        self.db_type = db_type
        self.db_precision = db_precision

        self.num_subprocesses = num_subprocesses
        if num_subprocesses < 1:
            logging.critical("number of sub processes should be at least"
                             " one, set to: {}".format(self.num_subprocesses))
            sys.exit()
        self.execution_mode = 'serial'
        if num_subprocesses > 1:
            self.execution_mode = 'multi-processing'

        self.mpi = mpi

    def upgrade(self) -> List:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.
        Returns list of missing parameters.
        """
        default_constr = OptimiserProperties()
        missing_attr = [x for x in default_constr.__dict__.keys()
                        if x not in self.__dict__.keys()]
        for x in missing_attr:
            self.__dict__[x] = default_constr.__dict__[x]
        return missing_attr

    def __str__(self) -> str:
        """
        Pretty print of (major) class values
        :return: string
        """

        ####
        # TODO: Revisit this attempt for backwards compatibility.
        if hasattr(self, 'backup_every'):
            be = self.backup_every
        else:
            be = 'Not available.'
        #####

        return "\noptimiser properties\n" \
               "--------------------\n" \
               "alg: {}\nrep: {}\nrep_backup: {}\nmax_loop_inc: {}\nngs: {}" \
               "\ndb_name: {}\ndb_type: {}" \
               "\ndb_precision: {}" \
               "\nexecution mode: {}" \
               "\nnumber of sub-processes: {}" \
               "\nmpi mode: {}\n".format(self.algorithm,
                                         self.repetitions,
                                         be,
                                         self.max_loop_inc,
                                         self.ngs,
                                         self.db_name,
                                         self.db_type,
                                         self.db_precision,
                                         self.execution_mode,
                                         self.num_subprocesses,
                                         self.mpi)


#################
# PARAMETER CLASS
class Parameter:
    """
    Stores general parameter values and meta data.
    This class is used for the parameters that the optimisation algorithm
    shall work with.
    Furthermore, it is used for meta data that could, for instance, describe
    the simulation environment (experimental conditions) but ARE NOT
    parameters that are optimised.
    """

    # TODO: Do None default values make sense?
    # TODO: Add type (f,e) of float output? How to deal with precision of 'f'?
    def __init__(self, name: str,
                 units: str = None,
                 place_holder: str = None,
                 value: Union[int, float] = None,
                 distribution: str = 'uniform',
                 min_value: float = None,
                 max_value: float = None,
                 max_increment: float = None,
                 output_float_precision: int = 6,
                 evaluate_value: str = None):
        """
        Constructor.
        :param name: name of parameter
        :param units: units as a string, default: None
        :param place_holder: place holder string used in templates, if not set,
            name is used
        :param value: holds current parameter value, which may also be the
            initial value
        :param distribution: parameter distribution function used for sampling,
            default: uniform, range: [uniform]
        :param min_value: assumed minimal value
        :param max_value: assumed maximal value
        :param max_increment: step size required for some optimisation
        algorithms
        :param output_float_precision: number of decimal positions places after
            the decimal sign for floats
        :param evaluate_value: string which contains the expression to be evaluated
            when replacing the placeholder
        """
        self.name = name
        self.units = units

        # set place holder to name, if not set
        if place_holder is not None:
            self.place_holder = place_holder
        else:
            self.place_holder = self.name

        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.distribution = distribution
        self.max_increment = max_increment
        self.output_float_precision = output_float_precision

        self.output_float_precision = output_float_precision

        self.evaluate_value = evaluate_value
        self.derived = False
        self.evaluated = None
        if self.evaluate_value is not None:
            self.derived = True
            self.evaluated = False

    def create_spotpy_parameter(self):
        pass

    def upgrade(self) -> list:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.
        Returns list of missing parameters.
        """
        default_constr = Parameter()
        missing_attr = [x for x in default_constr.__dict__.keys()
                        if x not in self.__dict__.keys()]
        for x in missing_attr:
            self.__dict__[x] = default_constr.__dict__[x]
        return missing_attr

    def __str__(self) -> str:
        """
        Creates string with parameter info.
        :return: info string
        """
        res = "name: {}".format(self.name)
        if self.units:
            res += ", units: {}".format(self.units)
        res += ", value: {}".format(self.value)
        if self.derived:
            res += ", evaluation string: {}".format(self.evaluate_value)
        return res


# TODO: add access elements via parameter name
class ParameterSet:
    """
    Container type for Parameter objects.
    """

    def __init__(self, name: str = None, params: List[Parameter] = None):
        """
        Constructor.
        :param name: optional label for parameter set
        :param params: initial list of parameters, deep copy is done
        """
        self.name = name
        self.parameters = []  # type: List[Parameter]

        # if set, deep copy the passed parameter into the self list
        if params:
            for p in params:
                # check for existing parameter name
                for tp in self.parameters:
                    if tp.name == p.name:
                        logging.error(
                            "Parameters with same names detected: {}".format(
                                p.name))
                        sys.exit(1)
                self.parameters.append(copy.deepcopy(p))

    def upgrade(self) -> List:
        """ Upgrade method updates object instance with default values,
            if pickle file is of older version.
            Returns list of missing parameters.
            !! CAREFUL !! Since lists like params will be init as [],
            it may cause unrecognised consequences.
        """
        default_constr = ParameterSet()
        missing_attr = [x for x in default_constr.__dict__.keys()
                        if x not in self.__dict__.keys()]
        for x in missing_attr:
            self.__dict__[x] = default_constr.__dict__[x]
        return missing_attr

    def update(self, other: 'ParameterSet'):
        """
        Updates parameter set with given other set.
        :param other: set used for update
        :return:
        """

        for ot in other.parameters:
            for my in self.parameters:
                if my.name == ot.name:
                    my.value = ot.value

        self.evaluate_derived_parameters()

    def __len__(self) -> int:
        """
        Return the length of the parameter set.
        :return: length of parameter set.
        """
        return len(self.parameters)

    def append(self, p: Parameter):
        """
        Append a copy given Parameter object to parameter list.
        :param p: parameter to be appended
        :return:
        """

        # check for existing parameter name
        for tp in self.parameters:
            if tp.name == p.name:
                logging.error(
                    "Parameters with same names detected: {}".format(p.name))
                sys.exit(1)

        self.parameters.append(copy.deepcopy(p))

    def reset_derived_parameter(self):
        """
        Reset all derived parameters.
        """
        for p in self.parameters:
            if p.derived:
                p.evaluated = False

    def evaluate_derived_parameters(self):
        """
        Evaluate all derived parameters.
        """

        # loop over all parameters to be derived, as long as there is progress, i.e. progress=True
        # if progress is false and parameters are not evaluated, not all dependencies have been
        # resolved

        self.reset_derived_parameter()

        progress = True
        while progress:
            progress = False
            # dumb loop over all parameters in this set
            for p in self.parameters:

                # consider only derived parameters and not yet evaluated ones
                if not p.derived:
                    continue
                elif p.evaluated:
                    continue

                # this dict will contain the parameters needed to evaluate the expression
                local_vars = {}

                print(f'Evaluating variable {p.name}, as {p.evaluate_value}')
                code = ast.parse(p.evaluate_value)

                skip_evaluation = False
                for node in ast.walk(code):
                    if type(node) is ast.Name:
                        var_name = node.id
                        print(f'  found required variable {var_name}')
                        p_index = self.get_index_by_name(var_name)
                        if p_index is None:
                            # Raise ERROR
                            print(f"ERROR, required parameter for variable {var_name} not found")
                            sys.exit(1)
                            return None
                        cp = self.parameters[p_index]
                        print(f'  found matching parameter {cp}')
                        if cp.derived and not cp.evaluated:
                            print(f'   skipping not evalued parameter {cp}')
                            skip_evaluation = True
                        local_vars[var_name] = self.parameters[p_index].value

                if not skip_evaluation:
                    result = eval(p.evaluate_value, local_vars)
                    print(f'  resulting value {result}')
                    p.value = float(result)
                    p.evaluated = True
                    progress = True

        for p in self.parameters:
            if p.evaluated == False:
                # Raise ERROR
                print(f"ERROR, not all parameters could be evaluated, one of them is {p}")
                sys.exit(1)

        return None


    def get_index_by_name(self, name: str):
        """
        Returns parameter index which matches a given name.

        :param name: selects the index of the chosen Parameter
        :return: selected Parameter object
        """

        for i in range(len(self.parameters)):
            if self.parameters[i].name == name:
                return i

        return None

    def __getitem__(self, item: int) -> Parameter:
        """
        Returns parameter at given index.
        :param item: selects the index of the chosen Parameter
        :return: selected Parameter object
        """
        return self.parameters[item]

    def __str__(self):
        """
        Pretty print of parameter set.
        :return: string with information about the set
        """
        res = "\n"
        head_line = "parameter set"
        if self.name:
            head_line += " ({})".format(self.name)
        res += head_line + "\n"
        res += len(head_line) * "-" + "\n"
        for s in self.parameters:
            res += str(s) + "\n"
        res += "\n"
        return res


# tests for Parameter classes
def test_parameter_setup():
    ps = ParameterSet("test set")
    ps.append(Parameter("density"))
    ps.append(Parameter("heat_flux"))

    print(ps)

def evaluate_parameters_test():
    ps = ParameterSet("evaluation test")
    ps.append(Parameter("density", value=5.6))
    ps.append(Parameter("heat_flux", value=25))
    ps.append(Parameter("my_value", evaluate_value='my_fraction ** 2'))
    ps.append(Parameter("my_fraction", evaluate_value='density * heat_flux'))

    ps.evaluate_derived_parameters()

    print(ps)

##################
# RELATION CLASSES

class DataSource:
    """
    Container for data and meta data of a data source, i.e. model or
    experimental data.
    """

    # TODO: add arguments to constructor
    # TODO: move read data to this class
    # TODO: allow for column index definition as alternative to labels
    def __init__(self):
        self.file_name = None
        self.header_line = None
        self.label_x = None
        self.label_y = None
        self.label_y2 = None
        # self.column_x = None
        # self.column_y = None
        self.x = None
        self.y = None
        self.y2 = None
        self.xfactor = 1.0
        self.xoffset = 0.0
        self.yfactor = 1.0
        self.yoffset = 0.0

        """
        :param file_name: file name which contains the information
        :param header_line: row that contains the labels (pandas data frames)
        :param label_x: label of the row which contains the information of the
            x-axis (pandas data frames)
        :param label_y:label of the row which contains the information of the
            y-axis (pandas data frames)
        :param x: data of the x-axis (based on above label)
        :param y: data of the y-axis (based on above label)
        :param factor:
        :param offset:
        """


class Relation:
    """
    Class representing a single relation between an experimental and model data
    set.
    """

    def __init__(self,
                 id_label: str = None,
                 model: DataSource = None,
                 experiment: DataSource = None,
                 fitness_method: FitnessMethodInterface = None,
                 fitness_weight: float = 1.0):
        """
        Set up a relation between the model and experiment data sources.
        :param id_label: a label entered by the user to better identify the
            relation
        :param x_def: definition range for both sources
        :param model: model data source
        :param experiment: experiment data source
        :param fitness_method: set fitness method
        :param fitness_weight:
        """

        self.id_label = id_label
        self.model = model if model else DataSource()
        self.experiment = experiment if experiment else DataSource()
        self.fitness_method = fitness_method
        self.x_e = None
        self.y_e = None
        self.fitness_weight = fitness_weight

    def read_data(self, wd: os.path, target: str = 'model'):
        """
        Read data from file and store it in the data source object.
        :param wd: working directory, i.e. directory prefix
        :param target: choose which data source is read, i.e. model or
            experimental source
        :return: None
        """

        # set ds to none and if it stays none, something went wrong
        ds = None
        if target == 'model':
            ds = self.model
        if target == 'experiment':
            ds = self.experiment

        # if experimental data was explicitly set to None,
        # like in case of an explicit
        # threshold fitness method, return, as there is no data to be read
        if ds is None:
            return

        # error handling
        if ds is None:
            logging.error("* Wrong data read target: {}".format(target))
            sys.exit()

        # if file name is not specified, do not read from file, as data may
        # have been set directly to ds.x / ds.y
        if ds.file_name is None:
            logging.warning("* Skip reading data, no data file defined.")
            return

        msg = "* From read_data: Read in data file: {} in directory: {}"
        logging.debug(msg.format(ds.file_name, wd))

        # Construct the input file name.
        in_file = os.path.join(wd, ds.file_name)
        # Read data as Pandas DataFrame.
        data = pd.read_csv(in_file, header=ds.header_line)

        # Get all header labels from the data frame.
        headers = list(data)
        # Check if the header labels from the input match with existing headers.
        msg = "* Wrong header: '{}' not found in {}"
        if ds.label_x not in headers:
            logging.error(msg.format(ds.label_x, in_file))
            sys.exit()
        elif ds.label_y not in headers:
            logging.error(msg.format(ds.label_y, in_file))
            sys.exit()

        logging.debug("* Size of read data: {}".format(data.shape))
        logging.debug("* Last data values: x={}, y={}".format(
            data[ds.label_x].dropna().values[-1],
            data[ds.label_y].dropna().values[-1]))

        # Get all header labels from the data frame.
        headers = list(data)
        # Check if the header labels from the input match with existing headers.
        msg = "* Wrong header: '{}' not found in {}"
        if ds.label_x not in headers:
            logging.error(msg.format(ds.label_x, in_file))
            sys.exit()
        elif ds.label_y not in headers:
            logging.error(msg.format(ds.label_y, in_file))
            sys.exit()

        logging.debug("* Size of read data: {}".format(data.shape))
        logging.debug("* Last data values: x={}, y={}".format(
            data[ds.label_x].dropna().values[-1],
            data[ds.label_y].dropna().values[-1]))

        # Assign data from file to data source arrays.
        ds.x = data[ds.label_x].dropna().values * ds.xfactor + ds.xoffset
        ds.y = data[ds.label_y].dropna().values * ds.yfactor + ds.yoffset

        msg = "* From read_data: ds.x={} ds.y={}"
        logging.debug(msg.format(ds.x, ds.y))

        if ds.label_y2 is not None:
            ds.y2 = data[ds.label_y2].dropna().values * ds.yfactor + ds.yoffset

    def compute_fitness(self):

        logging.debug("* Compute fitness")

        # error handling
        if self.fitness_method is None:
            logging.error("Specify fitness method!")
            sys.exit()

        ds_m = self.model
        ds_e = self.experiment

        # Debug information to check length of model response and
        # experimental data.
        logging.debug("* Model data: {}".format(ds_m))
        logging.debug("* Experiment data: {}".format(ds_e))

        # handle cases in which there is no experimental data set
        if ds_e is None:
            ds_e_x = None
            ds_e_y = None
            ds_e_y2 = None
        else:
            ds_e_x = ds_e.x
            ds_e_y = ds_e.y
            ds_e_y2 = None
            if ds_e.y2 is not None:
                ds_e_y2 = ds_e.y2

        fitness_value = self.fitness_method.compute(ds_e_x, ds_e_y,
                                                    ds_e_y2, ds_m.x,
                                                    ds_m.y)

        logging.debug("* Fitness value after compute: {}".format(fitness_value))

        return fitness_value

    def __str__(self) -> str:
        """
        Creates a string with the major relation information.
        :return: information string
        """
        res = "model file: {},\n" \
              "model header: {},\n" \
              "model x-label: {},\n" \
              "model y-label: {},\n" \
              "experiment file: {},\n" \
              "experiment header: {},\n" \
              "experiment x-label: {},\n" \
              "experiment y-label: {}\n" \
              "".format(self.model.file_name,
                        self.model.header_line,
                        self.model.label_x,
                        self.model.label_y,
                        self.experiment.file_name,
                        self.experiment.header_line,
                        self.experiment.label_x,
                        self.experiment.label_y)

        return res


# test for data read-in
def test_read_map_data():
    r = Relation()
    ds = r.model
    ds.file_name = 'TEST_devc.csv'
    ds.header_line = 1
    ds.label_x = 'Time'
    ds.label_y = 'VELO'

    ds = r.model
    ds.file_name = 'TEST_devc.csv'
    ds.header_line = 1
    ds.label_x = 'Time'
    ds.label_y = 'TEMP'

    r.read_data('test_data')
    r.x_def = r.model.x[::5]
    res = r.map_to_def()
    print(r.x_def, res)


#################
# EVALUATION METHOD CLASS
class EvaluationMethod:
    """
    Stores general parameter values and meta data.
    This class is used for the parameters that the optimisation algorithm
    shall work with.
    Furthermore, it is used for meta data that could, for instance, describe
    the simulation environment (experimental conditions) but ARE NOT
    parameters that are optimised.
    """

    def __init__(self, name: str = "Eval_01"):
        pass


#################
# FITNESS METHOD CLASS
class FitnessMethod:
    """
    Stores general parameter values and meta data.
    This class is used for the parameters that the optimisation algorithm
    shall work with.
    Furthermore, it is used for meta data that could, for instance, describe
    the simulation environment (experimental conditions) but ARE NOT
    parameters that are optimised.
    """

    def __init__(self, name: str = "RMSE"):
        pass


########################
# SIMULATION SETUP CLASS
class SimulationSetup:
    """
    A simulation setup is a collection of information to perform one
    optimisation run. Suppose you have three experiments, you would set up
    three simulation setups, e.g. TGA, DSC and Cone Calorimeter.
    They are collected later to form a SimulationSetupSet.
    """

    def __init__(self,
                 name: str,
                 work_dir: os.path = os.path.join('.'),
                 model_template: os.path = None,
                 model_input_file: os.path = 'model_input.file',
                 model_parameter: ParameterSet = None,
                 model_executable: os.path = None,
                 execution_dir: os.path = None,
                 execution_dir_prefix: os.path = None,
                 # best_dir: os.path='best_para',
                 analyser_input_file: os.path = 'input_analyser.py',
                 relations: List[Relation] = None,
                 evaluation_method: EvaluationMethod = None,
                 fitness_method: FitnessMethod = None):
        """
        Constructor.
        :param name: name for simulation setup
        :param work_dir: work directory, will contain all needed data
        :param model_template: points to the model input template
        :param model_input_file: name for model input file
        :param model_parameter: parameter set needed for this setup
        :param model_executable: call to invoke the model
        :param execution_dir: directory where the model execution will be
            carried out, mostly in temporally created directories
        :param best_dir: directory for performing simulation(s) with the best
            parameter set
        :param analyser_input_file: name for analyser input file
        :param relations: relations between experimental and model data
        :param evaluation_method:
        :param fitness_method:
        """

        self.name = name
        self.work_dir = work_dir
        self.model_template = model_template
        self.model_input_file = model_input_file
        self.model_parameter = model_parameter
        if model_parameter is None:
            self.model_parameter = ParameterSet()
        else:
            self.model_parameter = model_parameter
        self.model_executable = model_executable
        self.execution_dir = execution_dir
        self.execution_dir_prefix = execution_dir_prefix
        self.best_dir = os.path.join('Analysis', 'RunBestPara', self.name)
        self.analyser_input_file = analyser_input_file

        # if relations are set, check if a list is passed, otherwise create
        # a single element list
        if relations:
            if isinstance(relations, list):
                self.relations = relations
            else:
                self.relations = [relations]
        # if no value was passed, create an empty list
        else:
            self.relations = []

        self.id = None

    def upgrade(self) -> List:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.
        Returns list of missing parameters.
        """
        default_constr = SimulationSetup()
        missing_attr = [x for x in default_constr.__dict__.keys()
                        if x not in self.__dict__.keys()]
        for x in missing_attr:
            self.__dict__[x] = default_constr.__dict__[x]
        return missing_attr

    def __str__(self) -> str:
        """
        Creates a string with the major simulation setup information.
        :return: information string
        """
        res = "id: {}, name: {}, workdir: {}".format(self.id,
                                                     self.name,
                                                     self.work_dir)
        for p in self.model_parameter:
            res += "\n  " + str(p)

        return res


class SimulationSetupSet:
    """
    Container class for SimulationSetup objects.
    """

    def __init__(self,
                 name: str = None,
                 setups: List[SimulationSetup] = None):
        """
        Constructor.
        :param name: set name
        :param setups: list of initial setups
        """
        self.name = name

        # setups are passed, check if a single value was passed or a list
        if setups:
            if isinstance(setups, list):
                self.setups = setups
            else:
                self.setups = [setups]
        else:
            self.setups = []  # type: List[SimulationSetup]
        self.next_id = 0

    def upgrade(self) -> List:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.
        Returns list of missing parameters.
        !! Careful !! Since lists like SimulationSetupc will be init as [],
        it may cause unrecognised consequences.
        """
        default_constr = SimulationSetupSet()
        missing_attr = [x for x in default_constr.__dict__.keys()
                        if x not in self.__dict__.keys()]
        for x in missing_attr:
            self.__dict__[x] = default_constr.__dict__[x]
        return missing_attr

    def __len__(self) -> int:
        """
        Computes and returns the length of the set.
        :return: length of the simulation setup set
        """

        return len(self.setups)

    def append(self, s: SimulationSetup):
        """
        Appends a deep copy of the simulation setup to set.
        :param s: simulation setup to be appended
        :return: None
        """
        self.setups.append(copy.deepcopy(s))
        self.setups[-1].id = self.next_id
        self.next_id += 1

    def __getitem__(self, item: int) -> SimulationSetup:
        """
        Returns selected simulation setup.
        :param item: index of selected element
        :return: selected simulation setup
        """
        return self.setups[item]

    def __str__(self):
        """
        Creates an information string.
        :return: information string
        """
        res = "\n"
        head_line = "simulaton setup set"
        if self.name:
            head_line += " ({})".format(self.name)
        res += head_line + "\n"
        res += len(head_line) * "-" + "\n"
        for s in self.setups:
            res += str(s) + "\n"
        res += "\n"
        return res


class Version:
    """
    Version class to determine the current version of propti and simulation
    software in use.
    TODO : Think whether repr is the correct thing to code instead of str,i.e
    even though the class rep of the output variable is a 'Version' it does not
    represent a method by which the class could be initialized.
    """

    def __init__(self, setup):
        self.flag_propti = 0
        self.flag_exec = 0
        self.ver_propti = self.propti_versionCall()
        self.ver_exec = self.exec_versionCall(setup.model_executable)
        self.ver_spotpy = spotpy.__version__

    def propti_versionCall(self) -> str:
        """
        Look for propti-version and print a human readable representation.
        Print git hash value if no git is present.
        """
        # try:
        #     ver = subprocess.check_output(["git describe --always"
        #                                 ], shell=True).strip().decode("utf-8")
        # except subprocess.CalledProcessError as e:
        #     output = e.output
        #     self.flag_propti = e.returncode
        # # if git command doesn't exist
        #
        # if self.flag_propti != 0:  # TODO: This is a little hard coded(?)
        #     with open(os.path.join(script_location,
        #                            '../', '.git/refs/heads/master'), 'r') as f:
        #         ver = f.readline()[:7]
        #     with open(os.path.join(script_location,
        #                            '../', 'VERSION.txt'), 'r') as f:
        #         ver = f.readline()[6:24]
        #     f.close()

        try:
            ver = 'PROPTI-'
            with open(os.path.join(script_location,
                                   '../', 'VERSION.txt'), 'r') as f:
                ver += f.readline()[7:25]
            f.close()
            return ver
        except FileNotFoundError:
            self.flag_propti = 1
            return "Undetermined"

    def exec_versionCall(self, executable) -> str:
        """
        Look for executable version.
        Look for fds revision by calling fds without parameters
        and return its revision in use.
        # TODO: convert exec_versionCall completely to generic executable
        """
        try:
            # subprocess.check_call(['fds'], shell=True, stdout=subprocess.PIPE,
            #                    stderr=subprocess.PIPE)
            proc = subprocess.Popen([executable], shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)

            # TODO: make the parsing more general
            # This bit is only specific to FDS.

            # Define maximal number of line to be parsed.
            lines_count = 100
            while True:
                line = proc.stdout.readline().decode("utf-8")
                if line[1:9] == 'Revision':
                    ver = line[line.index(':') + 2:]
                    break
                lines_count -= 1
                if lines_count < 0:
                    self.flag_exec = 1
                    return "!! No Executable Present !!"
            return ver
        except subprocess.CalledProcessError:
            self.flag_exec = 1
            return "!! No Executable Present !!"

    def __repr__(self) -> str:
        string = self.ver_propti + ', ' + self.ver_exec
        return ('%r') % string

    def __str__(self) -> str:
        """
        Pretty print of class values
        :return: string
        """
        return "\nversion\n" \
               "--------------------\n" \
               "Propti Version: \t{}\n" \
               "Spotpy Version: \t{}\n" \
               "Executable Version:\t\t{}\n\n".format(self.ver_propti,
                                                      self.ver_spotpy,
                                                      self.ver_exec)


def test_simulation_setup_setup():
    sss = SimulationSetupSet("test setups")
    sss.append(SimulationSetup("cone1"))
    sss.append(SimulationSetup("tga2"))

    print(sss)


###############
# SAMPLER CLASS
class Sampler:
    """
    Stores sampler parameters and provides sampling schemes.
    """

    def __init__(self,
                 algorithm: str = 'LHS',
                 nsamples: int = 12,
                 deterministic: bool = False,
                 seed: int = None,
                 db_name: str = 'propti_db',
                 db_type: str = 'csv',
                 db_precision=np.float64):
        """
        Constructor.
        :param algorithm: choose sampling algorithm, default: LHS,
            range: [LHS]
        :param nsamples: number of samples, default: 12
        :param deterministic: If possible, use a deterministic sampling,
            default: false
        :param seed: If possible, set the seed for the random number generator, 
            default: None
        :param db_name: name of spotpy database file, default: propti_db
        :param db_type: type of database, default: csv, range: [csv]
        :param db_precision: desired precision of the values to be written into
            the data base, default: np.float64
        """
        self.algorithm = algorithm
        self.nsamples = nsamples
        self.deterministic = deterministic
        self.seed = seed
        self.db_name = db_name
        self.db_type = db_type
        self.db_precision = db_precision

    def __str__(self) -> str:
        """
        Pretty print of (major) class values
        :return: string
        """

        return "\nsampler properties\n" \
               "--------------------\n" \
               "alg: {}\nsamples: {}\ndeterministic: {}\nseed: {}" \
               "\ndb_name: {}\ndb_type: {}" \
               "\ndb_precision: {}\n".format(self.algorithm,
                                            self.nsamples,
                                            self.deterministic,
                                            self.seed,
                                            self.db_name,
                                            self.db_type,
                                            self.db_precision)

    def create_sample_set(self, params: ParameterSet) -> List[ParameterSet]:
        if self.algorithm == 'LHS':
            logging.info("Using LHS sampler")
            import scipy.stats



            bounds_low = []
            bounds_high = []
            param_name = []
            for p in params:
                print(p)
                if p.value is None:
                    print(  p.min_value, p.max_value)
                    bounds_low.append(p.min_value)
                    bounds_high.append(p.max_value)
                    param_name.append(p.name)

            sample_dim = len(bounds_low)

            sampler = scipy.stats.qmc.LatinHypercube(d = sample_dim)
            sample_raw = sampler.random(self.nsamples)
            sample_scaled = scipy.stats.qmc.scale(sample_raw, l_bounds=bounds_low, u_bounds=bounds_high)

            sampling_set = []
            sampling_index = 0
            for ps in sample_scaled:
                new_sample = ParameterSet(name=f"sample_{sampling_index:06d}", params=params)
                sampling_index += 1
                for ip in range(sample_dim):
                    new_sample[new_sample.get_index_by_name(param_name[ip])].value = ps[ip]
                sampling_set.append(new_sample)

            return sampling_set

        logging.critical("No maching sampler algorithm found.")


######
# MAIN

def data_structure_tests():
    # test_parameter_setup()
    # test_simulation_setup_setup()
    test_read_map_data()

# run tests if executed
if __name__ == "__main__":
    # data_structure_tests()
    evaluate_parameters_test()
