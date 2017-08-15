import os


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

class Relation:
    def __init__(self):
        self.model_x_label = None
        self.model_y_label = None
        self.experiment_x_label = None
        self.experiment_y_label = None
        self.x_def = None
        self.model_y = None
        self.experiment_y = None

class SimulationSetup:

    def __init__(self,
                 name: str,
                 work_dir: os.path = os.path.join('.'),
                 model_template: os.path = None,
                 model_input_file: os.path = 'model_input.file',
                 model_output_file: os.path = None,
                 model_parameter: ParameterSet = ParameterSet(),
                 model_executable: os.path = None,
                 execution_dir: os.path = None,
                 relationship_model_experiment: [Relation] = []):

        self.name = name
        self.work_dir = work_dir
        self.model_template = model_template
        self.model_input_file = model_input_file
        self.model_output_file = model_output_file
        self.model_parameter = model_parameter
        self.model_executable = model_executable
        self.exectuion_dir = execution_dir
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
    test_parameter_setup()
    test_simulation_setup_setup()

# run tests if executed
if __name__ == "__main__":
    data_structure_tests()
