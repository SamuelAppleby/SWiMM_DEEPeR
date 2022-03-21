@ECHO OFF
START python ..\..\test_env.py
START "" /d "..\..\..\UnityProjects\RLSimulation\Builds\Windows" ReinforcementLearningSimulation.exe server 127.0.0.1:60260
PAUSE
