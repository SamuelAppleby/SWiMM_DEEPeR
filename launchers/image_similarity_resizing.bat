set "PYTHONPATH=C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR;%PYTHONPATH%"
call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity_resizing.py --dir_orig C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\images --dir_unity_scaled C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\1920x1080\64x64 --num_samples 100
python image_similarity_resizing.py --dir_orig C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\images --dir_unity_scaled C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPeR\data\image_similarity\640x360\64x64 --num_samples 100
