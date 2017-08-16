import os
import numpy as np
import copy

from spotpy_wrapper import run_optimisation, create_input_file
from data_structures import Parameter, ParameterSet, \
    SimulationSetupSet, SimulationSetup, Relation, DataSource

CHID = 'CONE'
TEND = 600

op1 = Parameter(name='thickness1', place_holder='thickness1', min_value=1e-3,
               max_value=1e-1)

op2 = Parameter(name='thickness2', place_holder='thickness2', min_value=1e-3,
               max_value=1e-1)

ops = ParameterSet()
ops.append(op1)
ops.append(op2)

mp1 = Parameter(name='heat flux', place_holder='exflux', value=75)
mp2 = Parameter(name='tend', place_holder='tend', value=TEND)
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

r1 = Relation()
r1.model[0].file_name = "{}_hrr.csv".format(CHID)
r1.model[0].label_x = 'Time'
r1.model[0].label_y = 'MLR_TOTAL'
r1.model[0].header_line = 1
r1.experiment[0].file_name = "cone_example_01/Data.csv"
r1.experiment[0].label_x = '# Time_Alu_75'
r1.experiment[0].label_y = 'SG_Alu_75'
r1.experiment[0].header_line = 0
r1.experiment[0].factor = 1e-3

r1.read_data(wd='.', target='experiment')
r1.x_def = np.arange(0., TEND, 1)

s1 = SimulationSetup(name='cone_alu', work_dir='cone_alu',
                     model_template="cone_example_01/SimpleConeLaunchTest_Alu_BestParaSet_Toast.fds",
                     model_parameter=mps,
                     model_executable='fds',
                     relationship_model_experiment=[r1])

s2 = copy.deepcopy(s1)
s2.work_dir = 'cone_iso'
s2.model_template = "cone_example_01/SimpleConeLaunchTest_ISO_BestParaSet_Toast.fds"
s2.model_parameter[3].value = 575
s2.relationship_model_experiment[0].x_def = np.arange(0., s2.model_parameter[3].value, 1)
s2.relationship_model_experiment[0].experiment[0].label_x = "# Time_ISO_75"
s2.relationship_model_experiment[0].experiment[0].label_y = "SG_ISO_75"

setups = SimulationSetupSet()
setups.append(s1)
setups.append(s2)

for s in setups:
    if not os.path.exists(s.work_dir): os.mkdir(s.work_dir)

res = run_optimisation(ops, setups)

print(res)


