import sys
import os
import logging
import numpy as np
import pandas as pd


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

#################
# PARAMETER CLASS
class Parameter:
    def __init__(self, name: str,
                 units: str = None,
                 place_holder: str = None,
                 value: float = None,
                 min_value: float = None,
                 max_value: float = None,
                 distribution: str = None,
                 max_increment: float = None):

        self.name = name
        self.place_holder = place_holder
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.distribution = distribution
        self.max_increment = max_increment

    def __str__(self) -> str:
        return "name: {}, value: {}".format(self.name, self.value)

#TODO: add access elements via parameter name
class ParameterSet:
    def __init__(self, name: str = None):
        self.name = name
        self.parameters = []  # type: [Parameter]

    def __len__(self) -> int:
        return len(self.parameters)

    def append(self, p: Parameter):
        self.parameters.append(p)

    def __getitem__(self, item: int) -> Parameter:
        return self.parameters[item]

    def __setitem__(self, key: int, value: Parameter):
        self.parameters[key] = value

    def __str__(self):
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


def test_parameter_setup():

    ps = ParameterSet("test set")
    ps.append(Parameter("density"))
    ps.append(Parameter("heat_flux"))

    print(ps)


########################
# SIMULATION SETUP CLASS

class DataSource:
    def __init__(self):
        self.file_name = None
        self.header_line = None
        self.label_x = None
        self.label_y = None
        # self.column_x = None
        # self.column_y = None
        self.x = None
        self.y = None


class Relation:
    def __init__(self, x_def: np.ndarray = None, n_model: int = 1,
                 n_experiment: int = 1):
        # TODO: check that ns are at least one
        self.model = [] # type: list[DataSource]
        for i in range(n_model): self.model.append(DataSource())

        self.experiment = [] # type: list[DataSource]
        for i in range(n_experiment): self.experiment.append(DataSource())

        self.x_def = x_def

    def read_data(self, wd: os.path, target: str = 'model'):
        ds_set = None
        if target == 'model':
            ds_set = self.model
        if target == 'experiment':
            ds_set = self.experiment

        if ds_set is None:
            logging.error("wrong data read target: {}".format(target))

        for ds in ds_set:
            if ds.file_name is None: continue
            logging.debug("read in data file: {} in directory".format(ds.file_name, wd))
            in_file = os.path.join(wd, ds.file_name)
            data = pd.read_csv(in_file, header=ds.header_line)
            ds.x = data[ds.label_x]
            ds.y = data[ds.label_y]

    def map_to_def(self, target: str = 'model', mode:str = 'average',
                   len_only: bool = False):
        ds_set = None
        if target == 'model':
            ds_set = self.model
        if target == 'experiment':
            ds_set = self.experiment
        if ds_set is None:
            logging.error("wrong data read target: {}".format(target))

        if mode == 'average':
            if len_only: return len(self.x_def)
            res = np.zeros_like(self.x_def)

            for ds in ds_set:
                tmp_y = np.interp(self.x_def, ds.x, ds.y)
                res += tmp_y
            res /= len(ds_set)
            return res

        logging.error("wrong data mapping mode: {}".format(mode))

def test_read_map_data():

    r = Relation(n_model=2)
    ds = r.model[0]
    ds.file_name = 'TEST_devc.csv'
    ds.header_line = 1
    ds.label_x = 'Time'
    ds.label_y = 'VELO'

    ds = r.model[1]
    ds.file_name = 'TEST_devc.csv'
    ds.header_line = 1
    ds.label_x = 'Time'
    ds.label_y = 'TEMP'

    r.read_data('test_data')
    r.x_def = r.model[0].x[::5]
    res = r.map_to_def()
    print(r.x_def, res)

class SimulationSetup:

    def __init__(self,
                 name: str,
                 work_dir: os.path = os.path.join('.'),
                 model_template: os.path = None,
                 model_input_file: os.path = 'model_input.file',
                 model_parameter: ParameterSet = ParameterSet(),
                 model_executable: os.path = None,
                 execution_dir: os.path = None,
                 relationship_model_experiment: [Relation] = []):

        self.name = name
        self.work_dir = work_dir
        self.model_template = model_template
        self.model_input_file = model_input_file
        self.model_parameter = model_parameter
        self.model_executable = model_executable
        self.execution_dir = execution_dir
        self.realationship_model_experiment = relationship_model_experiment

        self.id = None

    def __str__(self) -> str:
        return "id: {}, name: {}, workdir: {}".format(self.id,
                                                      self.name,
                                                      self.work_dir)


class SimulationSetupSet:

    def __init__(self, name: str = None):
        self.name = name
        self.setups = []  # type: [SimulationSetup]
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.setups)

    def append(self, s: SimulationSetup):
        self.setups.append(s)
        self.setups[-1].id = self.next_id
        self.next_id += 1

    def __getitem__(self, item: int) -> SimulationSetup:
        return self.setups[item]

    def __setitem__(self, key: int, value: SimulationSetup):
        self.setups[key] = value

    def __str__(self):
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
