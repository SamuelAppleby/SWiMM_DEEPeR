@ECHO OFF
START python3 ..\..\test_env.py %* 
START /d ..\..\..\UnityProjects\RLSimulation\Builds\Windows\ ReinforcementLearningSimulation.exe
PAUSE
