import os
import sys
import logging
import copy
import numpy as np
import pandas as pd
from typing import List


#################
# OPTIMISER CLASS
class OptimiserProperties:
    """
    Stores optimiser parameter.
    """
    def __init__(self,
                 algorithm: str = 'sceua',
                 repetitions: int = 1,
                 ngs: int = None,  # will be set to len(ops) during
                                   # propti_prepare, if no value is provided
                 db_name: str = 'propti_db',
                 db_type: str = 'csv'):
        """
        Constructor.

        :param algorithm: choose spotpy algorithm, default: sceua,
            range: [sceua]
        :param repetitions: number of sampling repetitions, default: 1
        :param ngs: number of complexes, if None then set to len(para),
            default: None
        :param db_name: name of spotpy database file, default: propti_db
        :param db_type: type of database, default: csv, range: [csv]
        """
        self.algorithm = algorithm
        self.repetitions = repetitions
        self.ngs = ngs
        self.db_name = db_name
        self.db_type = db_type

    def __str__(self) -> str:
        """
        Pretty print of (major) class values
        :return: string
        """
        return "\noptimiser properties\n" \
               "--------------------\n" \
               "alg: {}\nrep: {}\nngs: {}" \
               "\ndb_name: {}\ndb_type: {}\n".format(self.algorithm,
                                                     self.repetitions,
                                                     self.ngs,
                                                     self.db_name,
                                                     self.db_type)


#################
# PARAMETER CLASS
class Parameter:
    """
    Stores general parameter values and meta data.
    """
    def __init__(self, name: str,
                 units: str = None,
                 place_holder: str = None,
                 value: float = None,
                 distribution: str = 'uniform',
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
        self.distribution = distribution
        self.max_increment = max_increment

    def __str__(self) -> str:
        """
        Creates string with parameter info.
        :return: info string
        """
        res = "name: {}".format(self.name)
        if self.units:
            res += ", units: {}".format(self.units)
        res += ", value: {}".format(self.value)
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
                self.parameters.append(copy.deepcopy(p))

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
        self.parameters.append(copy.deepcopy(p))

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
        # self.column_x = None
        # self.column_y = None
        self.x = None
        self.y = None
        self.factor = 1.0
        self.offset = 0.0


class Relation:
    """
    Class representing a single relation between an experimental and model data
    set.
    """
    def __init__(self,
                 x_def: np.ndarray = None,
                 model: DataSource = DataSource(),
                 experiment: DataSource = DataSource()):
        """
        Set up a relation between the model and experiment data sources.

        :param x_def: definition range for both sources
        :param model: model data source
        :param experiment: experiment data source
        """
        self.model = model
        self.experiment = experiment
        self.x_def = x_def

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
            logging.error("wrong data read target: {}".format(target))
            sys.exit()

        # if file name is not specified, do not read from file, as data may
        # have been set directly to ds.x / ds.y
        if ds.file_name is None:
            logging.warning("skip reading data, no data file defined")
            return

        logging.debug("read in data file: {} in directory".format(ds.file_name,
                                                                  wd))

        # construct the input file name
        in_file = os.path.join(wd, ds.file_name)
        # read data
        data = pd.read_csv(in_file, header=ds.header_line)
        # assign data from file to data source arrays
        ds.x = data[ds.label_x].dropna().values
        ds.y = data[ds.label_y].dropna().values

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
            logging.error("wrong data read target: {}".format(target))
            sys.exit()

        # which mode?
        if mode == 'average':
            # if length is only required, return just the length of the
            # definition set
            if len_only:
                return len(self.x_def)

            # interpolate data on the definition set and return it
            return np.interp(self.x_def, ds.x,
                             ds.y) * ds.factor + ds.offset

        # wrong mode was chosen
        logging.error("wrong data mapping mode: {}".format(mode))
        sys.exit()


# test for data readin
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


########################
# SIMULATION SETUP CLASS

class SimulationSetup:
    def __init__(self,
                 name: str,
                 work_dir: os.path = os.path.join('.'),
                 model_template: os.path = None,
                 model_input_file: os.path = 'model_input.file',
                 model_parameter: ParameterSet = ParameterSet(),
                 model_executable: os.path = None,
                 execution_dir: os.path = None,
                 best_dir: os.path = 'best_para',
                 relations: List[Relation] = None):
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
        :param relations: relations between experimental and model data
        """

        self.name = name
        self.work_dir = work_dir
        self.model_template = model_template
        self.model_input_file = model_input_file
        self.model_parameter = model_parameter
        self.model_executable = model_executable
        self.execution_dir = execution_dir
        self.best_dir = best_dir

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
    Cointainer class for SimulationSetup objects.
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
