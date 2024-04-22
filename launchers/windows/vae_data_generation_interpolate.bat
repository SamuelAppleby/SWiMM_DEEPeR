@echo off
cd ..\..\builds\windows
.\SWiMM_DEEPeR.exe modeVAEGen vaeInterpolate dataDir ..\..\data\interpolation resolutions 64 64 seed 7
exit