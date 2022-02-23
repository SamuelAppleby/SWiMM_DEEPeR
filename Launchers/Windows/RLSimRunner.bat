@ECHO OFF
START python ..\..\python_server.py %* 
START /d "..\..\..\UnityProjects\RLSimulation\Builds" ReinforcementLearningSimulation.exe
PAUSE
