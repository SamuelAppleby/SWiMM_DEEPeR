#!/bin/bash

# the following suppresses all output to the terminal, comment out if needed
#exec > /dev/null 2>&1

# run script - the "$@"" represent all the arguments passed in the command line in as many strings
# "$*"" will do the same thing but put all arguments in one space separated string
python3 ../../test_env.py "$@" &

# haven't tested this line yet, need the right build format
../../../RLSimulation/Builds/Linux/UnderwaterSimulation.x86_64

read