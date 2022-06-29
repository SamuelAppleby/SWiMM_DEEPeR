@ECHO OFF
cd..
cd.. 
START python train.py
cd RLSimulation\Builds\Windows\
START ReinforcementLearningSimulation.exe server 127.0.0.1:60260