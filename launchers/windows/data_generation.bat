@echo off
cd ..\..\builds\windows
.\SWiMM_DEEPeR.exe -modeVAEGen -noiseType noisy -dataDir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\noisy_images -resolutions 64 64 -numImages 1000 -seed 149
exit