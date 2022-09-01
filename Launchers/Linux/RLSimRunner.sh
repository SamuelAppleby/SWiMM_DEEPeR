#!/bin/bash

# the following suppresses all output to the terminal, comment out if needed
#exec > /dev/null 2>&1

# run script - the "$@"" represent all the arguments passed in the command line in as many strings
# "$*"" will do the same thing but put all arguments in one space separated string
cd ..
cd ..
python3 test_env.py "$@" &

# haven't tested this line yet, need the right build format
cd RLSimulation/Builds/Linux
chmod +x ReinforcementLearningSimulation.x86_64
ReinforcementLearningSimulation.x86_64 server 127.0.0.1:60260

exit