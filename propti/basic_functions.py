# -*- coding: utf-8 -*-

import os
import tempfile
import copy
import sys
import subprocess
import logging
import queue
import threading
import numpy as np

from .data_structures import Parameter, ParameterSet, SimulationSetup, \
    SimulationSetupSet, Relation


#####################
# INPUT FILE HANDLING

def create_input_file(setup: SimulationSetup, work_dir='execution'):

    """

    :param setup: specification of  SimulationSetup on which to base the
        simulation run
    :param work_dir: flag to indicate if the regular execution of the function
        (in the sense of inverse modeling) is wanted or if only a simulation
        of the best parameter set is desired, range:['execution', 'best']
    :return: Saves a file that is read by the simulation software as input file
    """
    #
    # small test
    if work_dir == 'execution':
        wd = setup.execution_dir
    elif work_dir == 'best':
        wd = setup.best_dir
    #
    #

    # Log the set working directory
    logging.debug(wd)

    in_fn = setup.model_template
    template_content = read_template(in_fn)

    logging.debug(template_content)

    parameter_list = setup.model_parameter
    input_content = fill_place_holder(template_content, parameter_list)

    logging.debug(input_content)

    out_fn = os.path.join(wd, setup.model_input_file)

    write_input_file(input_content, out_fn)


def write_input_file(content: str, filename: os.path):
    """

    :param content: Information that shall be written into a file, expected
        to be string.
    :param filename: File name of the new file.
    :return: File written to specified location.
    """
    try:
        outfile = open(filename, 'w')
    except OSError as err:
        logging.error("error writing input file: {}".format(filename))
        sys.exit()

    outfile.write(content)


def fill_place_holder(tc: str, paras: ParameterSet) -> str:
    # TODO: check for place holder duplicates
    res = tc
    if paras is not None:
        for p in paras:
            if type(p.value) == float:
                res = res.replace("#" + p.place_holder + "#",
                                  "{:E}".format(p.value))
            else:
                res = res.replace("#" + p.place_holder + "#", str(p.value))
    else:
        logging.warning("using empty parameter set for place holder filling")

    return res


def read_template(filename: os.path) -> str:
    try:
        infile = open(filename, 'r')
    except OSError as err:
        logging.error("error reading template file: {}".format(filename))
        sys.exit()

    content = infile.read()
    return content


def test_read_replace_template():
    wd = 'tmp'
    if not os.path.exists(wd):
        os.mkdir(wd)
    s = SimulationSetup("reader test", work_dir=wd)
    s.model_template = os.path.join('.', 'templates', 'basic_01.fds')
    s.model_input_file = "filled_basic_01.fds"

    p1 = Parameter("chid", place_holder="filename", value="toast_brot")
    p2 = Parameter("i", place_holder="i", value=42)
    p3 = Parameter("xmin", place_holder="xmin", value=1.463e-6)
    s.model_parameter.append(p1)
    s.model_parameter.append(p2)
    s.model_parameter.append(p3)

    create_input_file(s)


def test_missing_template():
    s = SimulationSetup("reader test")
    s.model_template = os.path.join('.', 'templates', 'notexists_basic_01.fds')
    create_input_file(s)


#################
# MODEL EXECUTION


def run_simulations(setups: SimulationSetupSet,
                    num_subprocesses: int = 1,
                    best_para_run: bool=False):
    """
    Executes each given SimulationSetup.

    :param setups: set of simulation setups
    :param num_subprocesses: determines how many sub-processes are to be used
        to perform the calculation, should be more than or equal to 1,
        default: 1, range: [integers >= 1]
    :param best_para_run: flag to switch to simulating the best parameter set
    :return: None
    """
    if num_subprocesses == 1:
        logging.info('serial model execution started')
        for s in setups:
            logging.info('start execution of simulation setup: {}'
                         .format(s.name))
            run_simulation_serial(s, best_para_run)
    else:
        logging.info('multi process execution started')
        run_simulation_mp(setups, num_subprocesses)


def run_simulation_serial(setup: SimulationSetup,
                          best_para_run: bool = False):

    # TODO: check return status of execution

    if best_para_run is False:
        new_dir = setup.execution_dir
    else:
        new_dir = setup.best_dir

    exec_file = setup.model_executable
    in_file = setup.model_input_file
    log_file = open(os.path.join(new_dir, "execution.log"), "w")

    # cmd = 'cd {}; {} {}'.format(new_dir, exec_file, in_file)
    cmd = 'cd {} && {} {}'.format(new_dir, exec_file, in_file)
    # cmd = ["cd {}".format(new_dir), " {} {}".format(exec_file, in_file)]

    logging.debug("executing command: {}".format(cmd))

    subprocess.check_call(cmd, shell=True,
                          stdout=log_file, stderr=log_file)
    log_file.close()



def run_simulation_mp(setups: SimulationSetupSet, num_threads:int = 1):

    def do_work(work_item: SimulationSetup):
        print("processing {}".format(work_item.name))
        run_simulation_serial(work_item)

    def worker():
        while True:
            work_item = q.get()
            if work_item is None:
                break
            do_work(work_item)
            q.task_done()

    q = queue.Queue()
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for item in setups:
        q.put(item)

    # block until all tasks are done
    q.join()

    # stop workers
    for i in range(num_threads):
        q.put(None)
    for t in threads:
        t.join()


def test_execute_fds():
    wd = 'tmp'
    if not os.path.exists(wd):
        os.mkdir(wd)
    s = SimulationSetup(name='exec test', work_dir=wd, model_executable='fds',
                        model_input_file=os.path.join('..', 'templates',
                                                      'basic_02.fds'))
    run_simulation_serial(s)


###########################
# ANALYSE SIMULATION OUTPUT

def extract_simulation_data(setup: SimulationSetup):
    # TODO: this is not general, but specific for FDS, i.e. first
    # TODO: line contains units, second the quantities names

    logging.debug("execution directory: {}".format(setup.execution_dir))

    for r in setup.relations:
        r.read_data(setup.execution_dir)


def map_data(x_def, x, y):
    return np.interp(x_def, x, y)


def test_prepare_run_extract():
    r1 = Relation()
    r1.model_x_label = "Time"
    r1.model_y_label = "VELO"
    r1.x_def = np.linspace(3.0, 8.5, 20)

    r2 = copy.deepcopy(r1)
    r2.model_y_label = "TEMP"

    relations = [r1, r2]

    paras = ParameterSet()
    paras.append(Parameter('ambient temperature', place_holder='TMPA'))
    paras.append(Parameter('density', place_holder='RHO'))

    s0 = SimulationSetup(name='ambient run',
                         work_dir='setup',
                         model_template=os.path.join('templates',
                                                     'template_basic_03.fds'),
                         model_executable='fds',
                         relations=relations,
                         model_parameter=paras
                         )

    setups = SimulationSetupSet()
    isetup = 0
    for tmpa in [32.1, 36.7, 42.7, 44.1]:
        current_s = copy.deepcopy(s0)
        current_s.model_parameter[0].value = tmpa
        current_s.work_dir += '_{:02d}'.format(isetup)
        setups.append(current_s)
        isetup += 1

    print(setups)

    for s in setups:
        if not os.path.exists(s.work_dir): os.mkdir(s.work_dir)

    for s in setups:
        create_input_file(s)

    for s in setups:
        run_simulations(s)

    for s in setups:
        extract_simulation_data(s)
        for r in s.relations:
            print(r.x_def, r.model_y)


def test_extract_data():
    s = SimulationSetup('test read data')
    s.model_output_file = os.path.join('test_data', 'TEST_devc.csv')

    r1 = ['VELO', ["none", "none"]]
    r2 = ['TEMP', ["none", "none"]]

    s.relations = [r1, r2]

    res = extract_simulation_data(s)

    for r in res:
        print(r)


######
# MAIN

# run tests if executed
if __name__ == "__main__":
    # test_read_replace_template()
    # test_execute_fds()
    # test_missing_template()
    # test_extract_data()
    test_prepare_run_extract()
    pass
