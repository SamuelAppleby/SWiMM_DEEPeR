@ECHO OFF
START python ..\..\server_training.py %* 
START /d ..\..\..\UnityProjects\RLSimulation\Builds\Windows\ ReinforcementLearningSimulation.exe
PAUSE