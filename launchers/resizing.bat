call ..\.venv\Scripts\activate
cd ..\benchmarking\scripts
python image_similarity.py --dir_orig ..\..\data\image_validation\1920x1080\images --dir_unity_scaled ..\..\data\image_validation\1920x1080\64x64 --dir_output ..\..\benchmarking\results --resize --num_samples 1000
python image_similarity.py --dir_orig ..\..\data\image_validation\640x360\images --dir_unity_scaled ..\..\data\image_validation\640x360\64x64 --dir_output ..\..\benchmarking\results --resize --num_samples 1000
