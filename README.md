## Process

![](propto.svg)

## Idea

Propti (working title) is an interface tool that couples FDS (https://github.com/firemodels/fds) with spotpy (https://github.com/thouska/spotpy/) to solve the inverse problem of material parameter estimation in a modular way. Propti covers setting up the general framework of the process as well as the preprocessing, e.g. generating FDS input files from (provided or self-made) templates and handing them to spotpy, and the postprocessing, e.g. analysing the evaluations of spotpy, find the best solution, run the best fitting parameters and plot all the data in nice figures.

The benefits of this interface tool are a standardized and convenient way to solve the inverse problem of material parameter optimization with all available options in FDS and spotpy, accompanied with template simulation set ups, central results database, benchmark database, reference input files, modeling and experimental guides.

## Features

- coupling of FDS and spotpy
- automatized pre- and postprocessing
- well documented
- parallelizable (MPI, OpenMP, multiprocessing)
- templates and reference input files
- suitable for HPC use
    - stop and restart processes
    - ...
- guide for optimization (including experimental design?)
- database for results and experimental data (?)
- benchmark database
- provide a platform to discuss modeling the problem
- ...

## Goals

### short term goals

- find new name
- implement framework and pre- and postprocessingfeatures
- provide templates and reference model designs
- provide a reliable base for propti by getting more involved in spotpy development and maintenance

### mid term goals

- promote use of Propti

### long term goals

- establish use of Propti

## ToDo

1. find a better name
2. collect currently used templates, strategies and code snippets
3. collect and evaluate requirements and needed features
4. define process and boundarys
6. define features
6. evaluate current templates, strategies and code snippets for reuse
6. implement
7. test
8. validate
8. use
9. evaluate
10. imporve
11. goto 'use'test

## Workshop

Date: 14.08. - 17.08. (tbc)
Place: Wuppertal (tbc)
