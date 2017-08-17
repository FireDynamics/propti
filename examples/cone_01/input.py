# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser

# import just for IDE convenience
import propti as pr

# fix the chid
CHID = 'CONE'

# define the optimisation parameter
op1 = pr.Parameter(name='thickness1', place_holder='thickness1',
                   min_value=1e-3, max_value=1e-1)
op2 = pr.Parameter(name='thickness2', place_holder='thickness2',
                   min_value=1e-3, max_value=1e-1)
ops = pr.ParameterSet(params=[op1, op2])

# define general model parameter, including optimisation parameter
mps0 = pr.ParameterSet(params=[op1, op2])
mps0.append(pr.Parameter(name='heat flux', place_holder='exflux', value=75))
mps0.append(pr.Parameter(name='tend', place_holder='tend'))
mps0.append(pr.Parameter(name='mesh_i', place_holder='i', value=3))
mps0.append(pr.Parameter(name='mesh_j', place_holder='j', value=3))
mps0.append(pr.Parameter(name='mesh_k', place_holder='k', value=4))
mps0.append(pr.Parameter(name='chid', place_holder='filename', value=CHID))

# define empty simulation setup set
setups = pr.SimulationSetupSet()

# loop over all 'iso' values
for iso in ['Alu', 'ISO']:

    # define model-experiment data relation
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

    # use above model prototype (mps0) as template
    mps = copy.deepcopy(mps0)

    TEND = 600
    # modify a single value of model parameter
    mps[3].value = TEND

    # define definition set for data comparison
    r.x_def = np.arange(0., TEND, 1)

    # create simulation setup object
    template_file = "SimpleConeLaunchTest_{}_BestParaSet_Toast.fds".format(iso)
    s = pr.SimulationSetup(name='cone_{}'.format(iso),
                           work_dir='cone_{}'.format(iso),
                           model_template=template_file,
                           model_parameter=mps,
                           model_executable='fds',
                           relationship_model_experiment=[r])

    # append above object to simulation setup set
    setups.append(s)

# use default values for optimiser
optimiser = pr.OptimiserProperties()