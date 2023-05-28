CALL .\..\.venv\Scripts\activate.bat
cd ..\benchmarking\scripts\gym_underwater
python image_similarity.py --dir_orig C:\Users\sambu\Downloads\images_high --dir_unity_scaled C:\Users\sambu\Downloads\images_high_scaled --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPer\benchmarking\results --resize --num_samples 1000
python image_similarity.py --dir_orig C:\Users\sambu\Downloads\images_low --dir_unity_scaled C:\Users\sambu\Downloads\images_low_scaled --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPer\benchmarking\results --resize --num_samples 1000
