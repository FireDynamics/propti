# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser

import propti as pr

CHID = 'CONE'

op1 = pr.Parameter(name='thickness1', place_holder='thickness1',
                   min_value=1e-3, max_value=1e-1)

op2 = pr.Parameter(name='thickness2', place_holder='thickness2',
                   min_value=1e-3, max_value=1e-1)

ops = pr.ParameterSet()
ops.append(op1)
ops.append(op2)

mp1 = pr.Parameter(name='heat flux', place_holder='exflux', value=75)
mp2 = pr.Parameter(name='tend', place_holder='tend')
mp3 = pr.Parameter(name='mesh_i', place_holder='i', value=3)
mp4 = pr.Parameter(name='mesh_j', place_holder='j', value=3)
mp5 = pr.Parameter(name='mesh_k', place_holder='k', value=4)
mp6 = pr.Parameter(name='chid', place_holder='filename', value=CHID)

mps0 = pr.ParameterSet()
mps0.append(op1)
mps0.append(op2)
mps0.append(mp1)
mps0.append(mp2)
mps0.append(mp3)
mps0.append(mp4)
mps0.append(mp5)
mps0.append(mp6)

setups = pr.SimulationSetupSet()
for iso in ['Alu', 'ISO']:

    r = pr.Relation()
    r.model.file_name = "{}_hrr.csv".format(CHID)
    r.model.label_x = 'Time'
    r.model.label_y = 'MLR_TOTAL'
    r.model.header_line = 1
    r.experiment.file_name = "Data.csv"
    r.experiment.label_x = '# Time_{}_75'.format(iso)
    r.experiment.label_y = 'SG_{}_75'.format(iso)
    r.experiment.header_line = 0
    r.experiment.factor = 1e-3

    mps = copy.deepcopy(mps0)

    # r.read_data(wd='.', target='experiment')
    TEND = 600 # r.experiment.x[-1]
    # print('tend: {}'.format(TEND))
    mps[3].value = TEND

    r.x_def = np.arange(0., TEND, 1)

    template_file = "SimpleConeLaunchTest_{}_BestParaSet_Toast.fds".format(iso)
    s = pr.SimulationSetup(name='cone_{}'.format(iso),
                           work_dir='cone_{}'.format(iso),
                           model_template=template_file,
                           model_parameter=mps,
                           model_executable='fds',
                           relationship_model_experiment=[r])

    setups.append(s)

optimiser = pr.OptimiserProperties()