import logging
import sys
import os
import shutil
import numpy as np
import tempfile

import spotpy

from .data_structures import Parameter, ParameterSet, SimulationSetup, \
    SimulationSetupSet, Relation, OptimiserProperties

from .basic_functions import create_input_file, run_simulations, \
    extract_simulation_data


####################
# SPOTPY SETUP CLASS

class SpotpySetup(object):
    def __init__(self,
                 params: ParameterSet,
                 setups: SimulationSetupSet,
                 optimiser: OptimiserProperties):

        self.setups = setups
        self.params = params
        self.optimiser = optimiser

        self.spotpy_parameter = []

        for p in params:
            logging.debug("setup spotpy parameter: {}".format(p.name))
            if p.distribution == 'uniform':

                optguess = None
                step = None
                if p.value is not None:
                    optguess = p.value
                if p.max_increment is not None:
                    step = p.max_increment

                cp = spotpy.parameter.Uniform(p.place_holder,
                                              p.min_value, p.max_value,
                                              step=step,
                                              optguess=optguess,
                                              minbound=p.min_value,
                                              maxbound=p.max_value)
                self.spotpy_parameter.append(cp)
            else:

                logging.error(
                    'parameter distribution function unknown: {}'.format(
                        p.distribution))

    def parameters(self):
        return spotpy.parameter.generate(self.spotpy_parameter)

    def simulation(self, vector):
        logging.debug("current spotpy simulation vector: {}".format(vector))

        # copyt spotpy parameter vector to parameter set
        for i in range(len(vector)):
            self.params[i].value = vector[i]

        # update all simulation setup parameter sets
        for s in self.setups:

            s.model_parameter.update(self.params)

        # create run directories for all simulation setups
        for s in self.setups:
            if s.execution_dir_prefix:
                tmp_dir_root = s.execution_dir_prefix
            else:
                tmp_dir_root = os.path.join(os.getcwd(), s.work_dir)
            s.execution_dir = tempfile.mkdtemp(prefix='rundir_',
                                               dir=tmp_dir_root)
            create_input_file(s)

        # run all simulatons
        run_simulations(self.setups, self.optimiser.num_subprocesses)

        # gather simulation data
        for s in self.setups:
            logging.debug("start data extraction")
            extract_simulation_data(s)

        # determine the length of all data sets
        n = 0
        for s in self.setups:
            for r in s.relations:
                n += r.map_to_def(len_only=True)

        res = np.zeros(n)
        index = 0
        for s in self.setups:
            for r in s.relations:
                n = r.map_to_def(len_only=True)
                res[index:index + n] = r.map_to_def()
                index += n

        for s in self.setups:
            shutil.rmtree(s.execution_dir)

        return res

    def evaluation(self):
        logging.debug("evaluation")
        for s in self.setups:
            for r in s.relations:
                r.read_data(wd='.', target='experiment')

        # determine the length of all data sets
        n = 0
        for s in self.setups:
            for r in s.relations:
                n += r.map_to_def(target='experiment', len_only=True)

        res = np.zeros(n)
        index = 0
        for s in self.setups:
            for r in s.relations:
                n = r.map_to_def(target='experiment', len_only=True)
                res[index:index + n] = r.map_to_def(target='experiment')
                index += n

        return res


def run_optimisation(params: ParameterSet,
                     setups: SimulationSetupSet,
                     opt: OptimiserProperties) -> ParameterSet:
    spot = SpotpySetup(params, setups, opt)

    if opt.algorithm == 'sceua':
        parallel = 'seq'
        if opt.mpi:
            parallel = 'mpi'
        sampler = spotpy.algorithms.sceua(spot,
                                          dbname=opt.db_name,
                                          dbformat=opt.db_type,
                                          alt_objfun='rmse',
                                          parallel=parallel)

        ngs = opt.ngs
        if not ngs:
            ngs = len(params)

            # Set amount of parameters as default for number of complexes
            # if not explicitly specified.
            opt.ngs = ngs

        sampler.sample(opt.repetitions, ngs=ngs)

        print(sampler.status.params)

    else:
        logging.critical("unknown algorithm set: {}".format(opt.algorithm))
        sys.exit()

    for i in range(len(params)):
        params[i].value = sampler.status.params[i]

    for s in setups:
        s.model_parameter.update(params)

    return params


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
    p1 = Parameter("ambient temperature", place_holder="TMPA", min_value=0,
                   max_value=100,
                   distribution='uniform', value=0)

    ps = ParameterSet()
    ps.append(p1)

    r1 = Relation()
    r1.model[0].label_x = "Time"
    r1.model[0].label_y = "TEMP"
    r1.model[0].file_name = 'TEST_devc.csv'
    r1.model[0].header_line = 1

    r1.experiment[0].x = np.linspace(0, 10, 20)
    r1.experiment[0].y = np.ones_like(r1.experiment[0].x) * 42.1

    r1.x_def = np.linspace(3.0, 8.5, 3)
    relations = [r1]

    s0 = SimulationSetup(name='ambient run',
                         work_dir='test_spotpy',
                         model_template=os.path.join('templates',
                                                     'template_basic_03.fds'),
                         model_executable='fds',
                         relations=relations,
                         model_parameter=ps
                         )
    setups = SimulationSetupSet()
    setups.append(s0)

    for s in setups:
        if not os.path.exists(s.work_dir):
            os.mkdir(s.work_dir)

    run_optimisation(ps, setups)


if __name__ == "__main__":
    # test_spotpy_setup()
    test_spotpy_run()
