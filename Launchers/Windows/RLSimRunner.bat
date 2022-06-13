@ECHO OFF
cd ..
cd ..
START python test_env.py
cd RLSimulation\Builds\Windows\
START ReinforcementLearningSimulation.exe server 127.0.0.1:60260
PAUSE
