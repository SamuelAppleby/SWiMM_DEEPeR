#!/bin/bash

# the following suppresses all output to the terminal, comment out if needed
exec > /dev/null 2>&1

# run script - the "$@"" represent all the arguments passed in the command line in as many strings
# "$*"" will do the same thing but put all arguments in one space separated string
python python_server.py "$@"

# haven't tested this line yet, need the right build format
../RLSimulation/Builds/ReinforcementLearningSimulation.x86_64

# haven't tested this yet, is meant to provide same functionality as pause, i.e. user press any key to continue
read –n1