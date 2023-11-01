@echo off
start "Training" cmd /c "train_sim.bat"
start "Tensor" cmd /c "tensor.bat"
cd ..\builds\windows
start SWiMM_DEEPeR.exe mode_server_control
exit