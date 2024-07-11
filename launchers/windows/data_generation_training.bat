@echo off
cd ..\..\builds\windows
.\SWiMM_DEEPeR.exe -modeVAEGen -dataDir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\cmvae_training_set -resolutions 64 64 -numImages 270000 -seed 3
exit