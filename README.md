

## Process

## Plan

Propti (working title) is an interface tool that couples FDS (https://github.com/firemodels/fds) with spotpy (https://github.com/thouska/spotpy/) to solve the inverse problem of material parameter estimation in a modular way. Propti covers setting up the general framework of the process as well as the preprocessing, e.g. generating FDS input files based on (provided or self-made) templates, handing them to spotpy, 

Propti makes available to use the pyrolysis model implemented in FDS, use different state-of-the-art optimization algorithms (modular), parallelization with MPI, OpenMP (FDS) and regular multiprocessing (FDS), automatized pre- and postprocessing, database of templates and reference designs, doing data- and file management, 

The benefits of this interface tool are a standardized way to realize   

## Features

- coupling of FDS and spotpy
- automatized pre- and postprocessing
- parallelizable (MPI, OpenMP, multiprocessing)
- templates and reference input files
- suitable for HPC use
    - stop and restart processes
    - ...
- guide for optimization (including experimental design?)
- database for results and experimental data (?)
- benchmark database
- 

## Goals

### Short term goals

- find new name
- getting more involved in spotpy development and maintenance to

### Mid term goals

- promote use of Propti

### Long term goals

- establish use of Propti

## ToDo

1. find a better name
2
3. collecting features in a mind map
4. discuss implementation 