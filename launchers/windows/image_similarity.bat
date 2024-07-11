@echo off
cd ..\..\builds\windows
.\SWiMM_DEEPeR.exe -modeVAEGen -dataDir C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity -resolutions 1920 1080 640 360 64 64 -numImages 1000 -seed 2
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\..\.venv\Scripts\activate
cd ..\..\benchmarking\scripts
python image_similarity_resizing.py --dirs_orig C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\images,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\images --dirs_scaled C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\64x64,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\64x64 --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\results
python image_similarity_cross_resolution.py --dirs C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\64x64,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\64x64,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\64x64\images --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\results
exit