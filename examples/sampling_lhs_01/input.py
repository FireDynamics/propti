# define variable 'params': sampling parameter set
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
params = pr.ParameterSet(params=[op1, op2, op3, op4])
params.append(pr.Parameter(name='chid', place_holder='CHID', value=CHID))

# define empty simulation setup set
setups = pr.SimulationSetupSet()

# create simulation setup object
template_file = "cone_template.fds"
s = pr.SimulationSetup(name='cone_pmma',
                       work_dir='cone_pmma',
                       execution_dir_prefix='samples_cone',
                       model_template=template_file,
                       model_parameter=params,
                       relations=None)

setups.append(s)

nsamples = 5
sampler = pr.Sampler(algorithm='LINEAR',
                     nsamples=nsamples)
time = []
for i in range(nsamples):
    time.append(f"0-00:1{i}:00")
job = pr.Job(template="fds", parameter=[
    ("CHID",CHID),
    ("TIME",time), 
    ("NODES","1")])