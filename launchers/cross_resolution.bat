CALL .\..\.venv\Scripts\activate.bat
cd ..\benchmarking\scripts
python3 image_similarity.py --dir_high_scaled C:\Users\sambu\Downloads\images_high_scaled --dir_low_scaled C:\Users\sambu\Downloads\images_low_scaled --dir_raw C:\Users\sambu\Downloads\images_raw --dir_output C:\Users\sambu\Documents\Repositories\CodeBases\SWiMM_DEEPer\benchmarking\results --cross_resolution --num_samples 1000