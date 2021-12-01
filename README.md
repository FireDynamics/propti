## Introduction

PROPTI is a Python module, written in Python 3.x. It provides a frame work for inverse modelling (or optimisation) of parameters in computer simulation. It's focused on handling the communication between simulation software and optimisation algorithms. Up to now, ["a Statistical Parameter Optimization Framework for Python" SPOTPY](https://github.com/thouska/spotpy) is used to provide a library of algorithm implementations. The [Fire Dynamics Simulator FDS](https://pages.nist.gov/fds-smv/) is used for the simulation side of things.

For newcomers to inverse modelling in fire (safety) simulation, PROPTI may serve as a good starting point. Specifically, its documentation and examples are aimed to provide a smooth entrance. Also, due to its generalised structure, input scripts can easily shared and discussed with collaborators. Furthermore, the open design of this framework allows it to be connected to arbitrary simulation software and/or algorithm libraries. Thus, PROPTI is not limited to fire safety engineering. Although, for now only SPOTPY and FDS connections are implemented, due to the current focus of the authors work.

It should also be noted: even though PROPTI started out for investigations of the simulation of material pyrolysis it is by NO MEANS limited to this application. This was just the focus of the primary authors at that time. PROPTI is very flexible and should be able to adjust nearly any input parameter and compare it to nearly any output. As long as the user can think of a way to combine the desired parameters it can be done! We have used PROPTI successfully to determine constituents of gas mixtures or to define RAMPs for fire using the HRRPUA, both obviously for FDS.

## Features

Input files, that are used to steer the inverse modelling process, are written in Python syntax. The user needs to provide templates for the simulation software input files and files of the experimental (target) data. PROPTI will collect all necessary files and group them in a separate directory. Meta-data is collected, as well and stored in a easy-to-use way for documentation and post processing purpose. Means to interact with the PROPTI framework via the command line are provided, even though its methods can of course be used in individually written Python scripts, as already known from the Python ecosystem.

Since the input file templates are text files, connection to arbitrary simulation software, which uses text input files, is relatively simple. Furthermore, the parameter set is generated, using the simulation software with which the actual simulation project is to be conducted, later on. Thus, the parameter set takes the limitations and advantages of said simulation software into account right from the start. However, this makes the parameter sets model specific.

Parallel execution of the algorithms is provided by the respective SPOTPY algorithms. Further parallelisation is provided within the PROPTI framework. Thus, it is relatively easy to set up inverse modelling processes across multiple simulation setups, for instance material parameter estimation based on different experiments.

Basic functionality for data analysis of the inverse modelling process is provided out of the box.

## Documentation and Examples

Documentation is provided in [Wiki](https://github.com/FireDynamics/propti/wiki). The folder 'examples' contains application examples tested with FDS version 6.7. 

## Citation

PROPTI is listed to ZENODO to get Data Object Identifiers (DOI) and allow for citations in scientific papers. You can find the necessary information here: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1188756.svg)](https://doi.org/10.5281/zenodo.1188756)

We have set up a project on ResearchGate: [PROPTI project](https://www.researchgate.net/project/PROPTI-An-Generalised-Inverse-Modelling-Framework)

Corresponding publications can be found here:

[PROPTI - A Generalised Inverse Modelling Framework](https://www.researchgate.net/publication/327655651_PROPTI_-_A_Generalised_Inverse_Modelling_Framework)

[Application cases of inverse modelling with the PROPTI framework](https://doi.org/10.1016/j.firesaf.2019.102835)

[Role of the Cost Function for Material Parameter Estimation](https://www.researchgate.net/publication/344217501_ROLE_OF_THE_COST_FUNCTION_FOR_MATERIAL_PARAMETER_ESTIMATION)
