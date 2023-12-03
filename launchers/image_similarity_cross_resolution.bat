call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity.py --dir_high_scaled ..\..\data\image_similarity\1920x1080\64x64 --dir_low_scaled ..\..\data\image_similarity\640x360\64x64 --dir_raw ..\..\data\image_similarity\64x64\images --cross_resolution --num_samples 1000