call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity.py --dir_high_scaled ..\..\data\image_validation\1920x1080\64x64 --dir_low_scaled ..\..\data\image_validation\640x360\64x64 --dir_raw ..\..\data\image_validation\64x64\images --dir_output ..\..\benchmarking\results --cross_resolution --num_samples 1000