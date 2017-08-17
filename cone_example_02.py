import os
import numpy as np
import copy
import pandas as pd

from spotpy_wrapper import run_optimisation, create_input_file
from data_structures import Parameter, ParameterSet, \
    SimulationSetupSet, SimulationSetup, Relation, DataSource

CHID = 'CONE'

op1 = Parameter(name='thickness1', place_holder='thickness1', min_value=1e-3,
               max_value=1e-1)

op2 = Parameter(name='thickness2', place_holder='thickness2', min_value=1e-3,
               max_value=1e-1)

ops = ParameterSet()
ops.append(op1)
ops.append(op2)

mp1 = Parameter(name='heat flux', place_holder='exflux', value=75)
mp2 = Parameter(name='tend', place_holder='tend')
mp3 = Parameter(name='mesh_i', place_holder='i', value=3)
mp4 = Parameter(name='mesh_j', place_holder='j', value=3)
mp5 = Parameter(name='mesh_k', place_holder='k', value=4)
mp6 = Parameter(name='chid', place_holder='filename', value=CHID)

mps = ParameterSet()
mps.append(op1)
mps.append(op2)
mps.append(mp1)
mps.append(mp2)
mps.append(mp3)
mps.append(mp4)
mps.append(mp5)
mps.append(mp6)

setups = SimulationSetupSet()
for iso in ['Alu', 'ISO']:

    r = Relation()
    r.model[0].file_name = "{}_hrr.csv".format(CHID)
    r.model[0].label_x = 'Time'
    r.model[0].label_y = 'MLR_TOTAL'
    r.model[0].header_line = 1
    r.experiment[0].file_name = "cone_example_01/Data.csv"
    r.experiment[0].label_x = '# Time_{}_75'.format(iso)
    r.experiment[0].label_y = 'SG_{}_75'.format(iso)
    r.experiment[0].header_line = 0
    r.experiment[0].factor = 1e-3

    r.read_data(wd='.', target='experiment')
    TEND = r.experiment[0].x[-1]
    # print('tend: {}'.format(TEND))
    mps[3].value = TEND

    r.x_def = np.arange(0., TEND, 1)

    s = SimulationSetup(name='cone_{}'.format(iso),
                        work_dir='cone_{}'.format(iso),
                        model_template="cone_example_01/SimpleConeLaunchTest_{}_BestParaSet_Toast.fds".format(iso),
                        model_parameter=mps,
                        model_executable='fds',
                        relationship_model_experiment=[r])

    setups.append(s)


for s in setups:
    if not os.path.exists(s.work_dir): os.mkdir(s.work_dir)

res = run_optimisation(ops, setups)

print(res)


