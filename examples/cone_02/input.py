# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser

# import just for IDE convenience
import propti as pr

# fix the chid
CHID = 'CONE'

# define the optimisation parameter
op1 = pr.Parameter(name='density', place_holder='DENSITY',
                   min_value=1e2, max_value=1e4)
op2 = pr.Parameter(name='emissivity', place_holder='EMISSIVITY',
                   min_value=0.01, max_value=1)
op3 = pr.Parameter(name='conductivity', place_holder='CONDUCTIVITY',
                   min_value=0.01, max_value=1)
op4 = pr.Parameter(name='specific_heat', place_holder='SPECIFIC_HEAT',
                   min_value=0.01, max_value=10)
ops = pr.ParameterSet(params=[op1, op2, op3, op4])

# define general model parameter, including optimisation parameter
mps = pr.ParameterSet(params=[op1, op2, op3, op4])
mps.append(pr.Parameter(name='chid', place_holder='CHID', value=CHID))

# define empty simulation setup set
setups = pr.SimulationSetupSet()

# define model-experiment data relation
r1 = pr.Relation()
r1.model.file_name = "{}_devc.csv".format(CHID)
r1.model.label_x = 'Time'
r1.model.label_y = 'temp'
r1.model.header_line = 1
r1.experiment.file_name = "experimental_data.csv"
r1.experiment.label_x = 'time'
r1.experiment.label_y = 'temp'
r1.experiment.header_line = 0
r1.fitness_method=pr.FitnessMethodRMSE(n_points=100)

r2 = pr.Relation()
r2.model.file_name = "{}_devc.csv".format(CHID)
r2.model.label_x = 'Time'
r2.model.label_y = 'temp'
r2.model.header_line = 1
r2.experiment = None
r2.fitness_method=pr.FitnessMethodThreshold("upper", threshold_target_value=90, threshold_value=400)

# create simulation setup object
template_file = "cone_template.fds"
s = pr.SimulationSetup(name='cone_pmma',
                       work_dir='cone_pmma',
                       model_template=template_file,
                       model_parameter=mps,
                       model_executable='fds',
                       relations=[r1,r2])

setups.append(s)

# use default values for optimiser
optimiser = pr.OptimiserProperties(algorithm='sceua',
                                   repetitions=10)
