import os
import sys
import logging
import copy
import numpy as np
import pandas as pd
import subprocess
import spotpy

from typing import List


# Reads the script's location. Used to access the PROPTI version number from the
# git repo.
script_location = os.path.dirname(os.path.realpath(__file__))


#################
# OPTIMISER CLASS
class OptimiserProperties:
    """
    Stores optimiser parameters. They describe parameters that are used by the
    optimiser, e.g. SPOTPY.
    """

    def __init__(self,
                 algorithm: str = 'sceua',
                 repetitions: int = 1,
                 backup_every: int = 100,
                 ngs: int = None,  # will be set to len(ops) during
                                   # propti_prepare, if no value is provided
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
        :param ngs: number of complexes, if None then set to len(para),
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
        self.ngs = ngs
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

    def upgrade(self) -> list:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.

        :return List of missing parameters.
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

        str_opt_prop = "\nOptimiser Properties\n" \
                       "--------------------\n" \
                       "Algorithm: {}\nRepetitions: {}\n" \
                       "Rep. before backup: {}\nNumber of complexes (ngs): {}" \
                       "\nData base name: {}\nData base type: {}" \
                       "\nDate base precision: {}" \
                       "\nExecution mode: {}" \
                       "\nNumber of sub-processes: {}" \
                       "\nMPI mode: {}\n".format(self.algorithm,
                                                 self.repetitions,
                                                 be,
                                                 self.ngs,
                                                 self.db_name,
                                                 self.db_type,
                                                 self.db_precision,
                                                 self.execution_mode,
                                                 self.num_subprocesses,
                                                 self.mpi)

        return str_opt_prop


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

    def __init__(self, name: str,
                 units: str = None,
                 place_holder: str = None,
                 value: float = None,
                 distribution: str = None,
                 min_value: float = None,
                 max_value: float = None,
                 max_increment: float = None):
        """
        Constructor.

        :param name: name of parameter
        :param units: units, default: None
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

        if distribution is None:
            self.distribution = 'uniform'
        else:
            self.distribution = distribution

        self.max_increment = max_increment

    def create_spotpy_parameter(self):
        pass

    def upgrade(self) -> list:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.

        :return list of missing parameter attributes.
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

        :return  info string
        """

        str_para = "\nParameter\n" \
                   "Name: {}\n" \
                   "--------------------\n" \
                   "Units: {}\nPlace holder: {}" \
                   "\nValue: {}\nDistribution: {}" \
                   "\nMinimum value: {}\nMaximum value: {}" \
                   "\nMaximum increment: {}\n" \
                   "".format(self.name, self.units,
                             self.place_holder, self.value,
                             self.distribution,
                             self.min_value, self.max_value,
                             self.max_increment)

        return str_para


# TODO: add access elements via parameter name
########################
# PARAMETER SET CLASS
class ParameterSet:
    """
    Container type for Parameter objects.
    """

    def __init__(self, name: str = None, params: list = None):
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
                self.parameters.append(copy.deepcopy(p))

    def upgrade(self) -> list:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.

        :return List of missing parameters.

        !! CAREFUL !! Since lists, like params, will be init as [],
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
        """

        for ot in other.parameters:
            for my in self.parameters:
                if my.name == ot.name:
                    my.value = ot.value

    def __len__(self) -> int:
        """
        Return the length of the parameter set.

        :return: length of parameter set.
        """

        return len(self.parameters)

    def append(self, p: Parameter):
        """
        Append a copy of a given Parameter object to parameter list.

        :param p: parameter to be appended
        """

        self.parameters.append(copy.deepcopy(p))

    def __getitem__(self, item: int) -> Parameter:
        """
        Returns parameter at given index.

        :param item: selects the index of the chosen Parameter
        :return selected Parameter object
        """

        return self.parameters[item]

    def __str__(self):
        """
        Pretty print of parameter set.

        :return: string with information about the set
        """

        if self.name:
            psn = self.name
        else:
            psn = "Not available."

        res = "\nParameter Set\n" \
              "Name: {}\n" \
              "--------------------\n".format(psn)

        # Check if the list ParameterSet is empty. Otherwise provide
        # information on the contained parameters.
        if not self.parameters:
            res += "Parameter set is empty."
        else:
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


##################
# DATA SOURCE CLASS
class DataSource:
    """
    Container for data and meta data of a data source, i.e. model or
    experimental data.
    """

    # TODO: move read data to this class
    # TODO: allow for column index definition as alternative to labels
    def __init__(self, file_name: str = None,
                 header_line: int = None,
                 label_x: str = None,
                 label_y: str = None,
                 column_x: int = None,
                 column_y: int = None,
                 x_values: list = None,
                 y_values: list = None,
                 # TODO: clean the following lines up
                 # factor: float = 1.0,
                 # offset: float = 0.0):
                 factor: float = None,
                 offset: float = None):

        """
        Constructor.

        :param file_name: Name of the file which contains the desired data,
            simulation or experiment.
        :param header_line: Row that contains the column labels (like pandas
            data frames).
        :param label_x: Label of the column which contains the information of
            the x-axis (pandas data frames).
        :param label_y: Label of the column which contains the information of
            the y-axis (pandas data frames).
        :param column_x: Index of the column containing the data series of
            the x_values.
        :param column_y: Index of the column containing the data series of
            the y_values.
        :param x_values: Data of the x-axis (based on above label).
        :param y_values: Data of the y-axis (based on above label).
        :param factor: Factor to scale the data on-the-fly.
        :param offset: Offset to shift the data on-the-fly.
        """

        self.file_name = file_name
        self.header_line = header_line
        self.label_x = label_x
        self.label_y = label_y
        self.column_x = column_x
        self.column_y = column_y
        self.x_values = x_values
        self.y_values = y_values
        self.factor = factor
        self.offset = offset

        # Set default values for factor.
        if self.factor is None:
            self.factor = 1.0
        else:
            self.factor = factor

        # Set default values for offset.
        if self.offset is None:
            self.offset = 0.0
        else:
            self.offset = offset

    def __str__(self):
        """
        Pretty print of parameter set.

        :return string with information about the set
        """

        str_data_source = "\nData Source\n" \
                          "--------------------\n" \
                          "File name: {}\n" \
                          "Header line: {}\nLabel x-data: {}" \
                          "\nLabel x-data: {}\nColumn index x: {}" \
                          "\nColumn index y: {}\nFactor: {}" \
                          "\nOffset: {}\n" \
                          "".format(self.file_name, self.header_line,
                                    self.label_x, self.label_y,
                                    self.column_x, self.column_y,
                                    self.factor, self.offset)

        return str_data_source


########################
# RELATION CLASS
class Relation:
    """
    Class representing a single relation between an experimental and model data
    set.
    """

    def __init__(self,
                 x_def: np.ndarray = None,
                 model: DataSource = None,
                 experiment: DataSource = None):
        """
        Set up a relation between the model and experiment data sources.

        :param x_def: definition range for both sources
        :param model: model data source
        :param experiment: experiment data source
        """

        self.x_def = x_def
        self.model = model if model else DataSource()
        self.experiment = experiment if experiment else DataSource()

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

        # error handling
        if ds is None:
            logging.error("Wrong data read target: {}".format(target))
            sys.exit()

        # if file name is not specified, do not read from file, as data may
        # have been set directly to ds.x / ds.y
        if ds.file_name is None:
            logging.warning("* Skip reading data, no data file defined")
            return

        logging.debug("* Read in data file: {} in directory"
                      "".format(ds.file_name, wd))

        # construct the input file name
        in_file = os.path.join(wd, ds.file_name)
        # read data
        data = pd.read_csv(in_file, header=ds.header_line)
        # assign data from file to data source arrays
        ds.x_values = data[ds.label_x].dropna().values
        ds.y_values = data[ds.label_y].dropna().values

    def map_to_def(self,
                   target: str = 'model',
                   mode: str = 'average',
                   len_only: bool = False):
        """
        Maps the data of a data source to a definition set.

        :param target: choose data source, i.e. experiment or model,
            default: model, range: [model, experiment]
        :param mode: choose if data should be processed, default: average,
            range: [average]
        :param len_only: if set, return only the length of the resulting array,
            without creating the array
        :return: mapped data array or length of array
        """

        # set ds to none and if it stays none, something went wrong
        ds = None
        if target == 'model':
            ds = self.model
        if target == 'experiment':
            ds = self.experiment
        if ds is None:
            logging.error("* Wrong data read target: {}".format(target))
            sys.exit()

        # which mode?
        if mode == 'average':
            # if length is only required, return just the length of the
            # definition set
            if len_only:
                return len(self.x_def)

            # interpolate data on the definition set and return it
            return np.interp(self.x_def, ds.x_values,
                             ds.y_values) * ds.factor + ds.offset

        # wrong mode was chosen
        logging.error("* Wrong data mapping mode: {}".format(mode))
        sys.exit()

    def __str__(self) -> str:
        """
        Creates a string with the major relation information.

        :return: information string
        """

        str_relation = "\nRelation\n" \
                       "--------------------\n" \
                       "model file: {},\n" \
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

        return str_relation


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
    r.x_def = r.model.x_values[::5]
    res = r.map_to_def()
    print(r.x_def, res)


########################
# SIMULATION SETUP CLASS
class SimulationSetup:
    """
    A simulation setup is a collection of information to perform one
    optimisation run. Suppose you have three experiments, you would set up
    three simulation setups.
    They are collected later in a SimulationSetupSet.
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
                 best_dir: os.path = 'best_para',
                 analyser_input_file: os.path = 'input_analyser.py',
                 relations: List[Relation] = None):
        """
        Constructor.

        :param name: name for simulation setup
        :param work_dir: work directory, will contain all needed data
        :param model_template: points to the model input template
        :param model_input_file: name for model input file, i.e. the
            parameters that are worked on by the optimisation algorithm
            (NOT the environment!)
        :param model_parameter: parameter set needed for this setup
        :param model_executable: call to invoke the model
        :param execution_dir: Directory where the model execution will be
            carried out, mostly in temporally created directories in it.
            The sub-directory names get a random name to avoid conflicts
            during mpi.
        :param execution_dir_prefix: Identifier for the temp directories,
            to have means to connect them to simulation setups, if needed (error
            tracking).
        :param best_dir: directory for performing simulation(s) with the best
            parameter set
        :param analyser_input_file: name for analyser input file
        :param relations: relations between experimental and model data
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
        self.best_dir = best_dir
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

        str_sim_setup = "\nSimulation Setup\n" \
                        "Name: {},\n" \
                        "--------------------\n" \
                        "ID: {},\n" \
                        "Working directory: {},\n" \
                        "Model template: {},\n" \
                        "Model input file: {},\n" \
                        "Model executable: {},\n" \
                        "Execution directory: {},\n" \
                        "Execution dir. prefix: {},\n" \
                        "Best simulation dir.: {},\n" \
                        "Analyser input file: {},\n" \
                        "Relations: {}\n" \
                        "\nParameters of this simulation setup:\n" \
                        "--------------------" \
                        "".format(self.name,
                                  self.id,
                                  self.work_dir,
                                  self.model_template,
                                  self.model_input_file,
                                  self.model_executable,
                                  self.execution_dir,
                                  self.execution_dir_prefix,
                                  self.best_dir,
                                  self.analyser_input_file,
                                  self.relations)

        for p in self.model_parameter:
            str_sim_setup += "" + str(p)
            # str_sim_setup += "\n  " + str(p)

        return str_sim_setup


########################
# SIMULATION SETUP SET CLASS
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

    def upgrade(self) -> list:
        """
        Upgrade method updates object instance with default values,
        if pickle file is of older version.
        Returns list of missing parameters.
        !! Careful !! Since lists like SimulationSetup will be init as [],
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
        head_line = "Simulaton Setup Set"
        if self.name:
            head_line += " ({})".format(self.name)
        res += head_line + "\n"
        res += len(head_line) * "-" + "\n"
        for s in self.setups:
            res += str(s) + "\n"
        res += "\n"
        return res


########################
# VERSION CLASS
class Version:
    # TODO: clean-up this class!
    """
    Version class to determine the current version of PROPTI and simulation
    software in use.
    """

    def __init__(self):
        """
        :param flag_propti:
        :param flag_exec:
        :param ver_propti: Calls and stores the PROPTI version.
        :param ver_exec: Calls and stores the simulation software executable
            version.
        :param ver_spotpy: Calls and stores the SPOTPY version.
        """
        self.flag_propti = 0
        self.flag_exec = 0
        self.ver_propti = self.propti_version_call()
        self.ver_exec = self.exec_version_call()
        self.ver_spotpy = spotpy.__version__

    def propti_version_call(self) -> str:
        """
        Look for propti-version and print a human readable representation.
        Print git hash value if no git is present.

        :return: PROPTI version.
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

    def exec_version_call(self) -> str:
        """
        Look for executable version.
        Look for fds revision by calling fds without parameters
        and return its revision in use.

        :return: Version of the simulation software executable.
        """
        # TODO: convert exec_version_call completely to generic executable

        try:
            # subprocess.check_call(['fds'], shell=True, stdout=subprocess.PIPE,
            #                    stderr=subprocess.PIPE)
            proc = subprocess.Popen(['fds'], shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            # This bit is only specific to FDS
            while True:
                line = proc.stdout.readline().decode("utf-8")
                if line[1:9] == 'Revision':
                    ver = line[line.index(':')+2:]
                    break
            return ver
        except subprocess.CalledProcessError:
            self.flag_exec = 1
            return "!! No Executable Present !!"

    # TODO : Think whether repr is the correct thing to code instead of str,i.e
    # even though the class rep of the output variable is a 'Version' it does
    # not represent a method by which the class could be initialized.
    def __repr__(self) -> str:
        string = self.ver_propti + ', ' + self.ver_exec
        return ('%r') % string

    def __str__(self) -> str:
        """
        Pretty print of class values

        :return: string
        """

        str_version = "\nVersion\n" \
                      "--------------------\n" \
                      "PROPTI Version: \t{}\n" \
                      "SPOTPY Version: \t{}\n" \
                      "Executable Version:\t{}\n\n".format(self.ver_propti,
                                                           self.ver_spotpy,
                                                           self.ver_exec)

        return str_version


def test_simulation_setup_setup():
    sss = SimulationSetupSet("test setups")
    sss.append(SimulationSetup("cone1"))
    sss.append(SimulationSetup("tga2"))

    print(sss)


######
# MAIN

def data_structure_tests():
    # test_parameter_setup()
    # test_simulation_setup_setup()
    test_read_map_data()


# run tests if executed
if __name__ == "__main__":
    data_structure_tests()
