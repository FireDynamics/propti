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
r = pr.Relation()
r.model.file_name = "{}_devc.csv".format(CHID)
r.model.label_x = 'Time'
r.model.label_y = 'temp'
r.model.header_line = 1
r.experiment.file_name = "experimental_data.csv"
r.experiment.label_x = 'time'
r.experiment.label_y = 'temp'
r.experiment.header_line = 0
r.fitness_method=pr.FitnessMethodRMSE(n_points=100)

# create simulation setup object
template_file = "cone_template.fds"
s = pr.SimulationSetup(name='cone_pmma',
                       work_dir='cone_pmma',
                       model_template=template_file,
                       model_parameter=mps,
                       model_executable='fds',
                       relations=r)

setups.append(s)

# use default values for optimiser
optimiser = pr.OptimiserProperties(algorithm='sceua',
                                   repetitions=10)