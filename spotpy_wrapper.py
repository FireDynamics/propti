import logging
import sys
import os
import numpy as np

import spotpy

from data_structures import Parameter, ParameterSet, SimulationSetup, \
    SimulationSetupSet, Relation

from basic_functions import create_input_file, run_simulation, \
    extract_simulation_data

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

class SpotpySetup(object):
    def __init__(self, params: ParameterSet, setups: SimulationSetupSet):
        self.parameter = []

        self.setups = setups

        self.it = 0

        print(params)

        for p in params:
            logging.debug("setup spotpy parameter: {}".format(p.name))
            if p.distribution == 'uniform':
                cp = spotpy.parameter.Uniform(p.place_holder,
                                              p.min_value, p.max_value,
                                              step=p.max_increment,
                                              optguess=p.value,
                                              minbound=p.min_value,
                                              maxbound=p.max_value)
                self.parameter.append(cp)
            else:
                logging.error('parameter distribution function unkown: {}'.
                              format(p.distribution))
    def parameters(self):
        return spotpy.parameter.generate(self.parameter)

    def simulation(self, vector):
        logging.debug("current simulation vector: {}".format(vector))
        logging.debug("iteration: {}".format(self.it))

        for s in self.setups:
            s.model_parameter[0].value = vector[0]
            create_input_file(s)
            run_simulation(s)
            extract_simulation_data(s)

        self.it += 1
        return self.setups[0].realationship_model_experiment[0].model_y

    def evaluation(self):
        logging.debug("evaluation")
        return self.setups[0].realationship_model_experiment[0].experiment_y

def run_optimisation(params: ParameterSet,
                     setups: SimulationSetupSet) -> ParameterSet:

    spot = SpotpySetup(params, setups)

    sampler = spotpy.algorithms.sceua(spot,
                                      dbname='propty',
                                      dbformat='csv',
                                      alt_objfun='rmse')

    print("objective function: {}".format(sampler.objectivefunction))
    sampler.sample(10, ngs=len(params))

def test_spotpy_setup():

    p1 = Parameter("density", "RHO", min_value=1.0, max_value=2.4,
                   distribution='uniform')
    p2 = Parameter("cp", place_holder="CP", min_value=4.0, max_value=7.2,
                   distribution='uniform')

    ps = ParameterSet()
    ps.append(p1)
    ps.append(p2)

    spot = SpotpySetup(ps)

    for p in spot.parameter:
        print(p.name, p.rndargs)

def test_spotpy_run():
    p1 = Parameter("ambient temperature", place_holder="TMPA", min_value=0, max_value=100,
                   distribution='uniform', value=0)

    ps = ParameterSet()
    ps.append(p1)

    r1 = Relation()
    r1.model_x_label = "Time"
    r1.model_y_label = "TEMP"
    r1.x_def = np.linspace(3.0, 8.5, 3)
    r1.experiment_y = np.ones_like(r1.x_def) * 42.1

    relations = [r1]
    s0 = SimulationSetup(name='ambient run',
                         work_dir='test_spotpy',
                         model_template=os.path.join('templates',
                                                     'template_basic_03.fds'),
                         model_executable='fds',
                         model_output_file='TEST_devc.csv',
                         relationship_model_experiment=relations,
                         model_parameter=ps
                         )
    setups = SimulationSetupSet()
    setups.append(s0)

    for s in setups:
        if not os.path.exists(s.work_dir): os.mkdir(s.work_dir)

    run_optimisation(ps, setups)

if __name__ == "__main__":
    # test_spotpy_setup()
    test_spotpy_run()