@echo off
set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity_cross_resolution.py --dirs C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\64x64,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\64x64,C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\64x64\images --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\results