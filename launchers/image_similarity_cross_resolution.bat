set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity_cross_resolution.py --dir_high_scaled C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\64x64 --dir_low_scaled C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\64x64 --dir_raw C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\64x64\images --num_samples 100