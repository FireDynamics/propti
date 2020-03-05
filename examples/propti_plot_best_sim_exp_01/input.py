# define variable 'ops': optimisation parameter set
# define variable 'setups': simulation setup set
# define variable 'optimiser': properties for the optimiser

# import just for IDE convenience
import propti as pr

# fix the chid
CHID = 'TGA_analysis_01'
TEND = 9360


# use default values for optimiser
optimiser = pr.OptimiserProperties(algorithm='sceua',
                                   repetitions=10)
#ngs=4,

# define the optimisation parameter
op1 = pr.Parameter(name='ref_temp_comp_01',
                   place_holder='rtc01',
                   min_value=200, max_value=400)
op2 = pr.Parameter(name='ref_rate_comp_01',
                   place_holder='rrc01',
                   min_value=0.001, max_value=0.01)
op3 = pr.Parameter(name='ref_temp_comp_02',
                   place_holder='rtc02',
                   min_value=300, max_value=600)
op4 = pr.Parameter(name='ref_rate_comp_02',
                   place_holder='rrc02',
                   min_value=0.001, max_value=0.01)

ops = pr.ParameterSet(params=[op1, op2, op3, op4])

# define general model parameter, including optimisation parameter
mps0 = pr.ParameterSet(params=[op1, op2, op3, op4])
mps0.append(pr.Parameter(name='heating rate', place_holder='hr', value=10))
mps0.append(pr.Parameter(name='chid', place_holder='CHID', value=CHID))

# define empty simulation setup set
setups = pr.SimulationSetupSet()

# define model-experiment data relation
r = pr.Relation()
r.model.file_name = "{}_tga.csv".format(CHID)
r.model.label_x = 'Time'
r.model.label_y = 'Total MLR'
r.model.header_line = 1
r.experiment.file_name = "tga_experimental_data.csv"
r.experiment.label_x = 'Time'
r.experiment.label_y = 'MassLossRate'
r.experiment.header_line = 0
r.fitness_method=pr.FitnessMethodRMSE(n_points=100)

# create simulation setup object
template_file = "tga_analysis_01.fds"
s = pr.SimulationSetup(name='tga_analysis_01',
                       work_dir='tga_analysis_run_01',
                       model_template=template_file,
                       model_parameter=mps0,
                       model_executable='fds',
                       relations=r)

# append above object to simulation setup set
setups.append(s)

