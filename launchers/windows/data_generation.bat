@echo off
cd ..\..\builds\windows
.\SWiMM_DEEPeR.exe -modeVAEGen -noiseType clean -dataDir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\noiseless_images_2 -resolutions 64 64 -numImages 1000 -seed 149
exit