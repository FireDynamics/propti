## Process

![](resources/jureca/propti.png)

## Introduction

PROPTI is a Python module, written in Python 3.x. It provides a frame work for inverse modelling (or optimisation) of parameters in computer simulation. It's focused on handling the communication between simulation software and optimisation algorithms. Up to now, "a Statistical Parameter Optimization Framework for Python" SPOTPY is used to provide a library of algorithm implementations. The Fire Dynamics Simulator FDS is used for the simulation side of things.

For newcomers to inverse modelling in fire (safety) simulation, PROPTI may serve as a good starting point. Specifically, its documentation and examples are aimed to provide a smooth entrance. Also, due to its generalised structure, input scripts can easily shared and discussed with collaborators. Furthermore, the open design of this framework allows it to be connected to arbitrary simulation software and/or algorithm libraries. Thus, PROPTI is not limited to fire safety engineering. Although, for now only SPOTPY and FDS connections are implemented, due to the current focus of the authors work.
