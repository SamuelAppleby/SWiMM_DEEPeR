cd ../../builds/windows
SWiMM_DEEPeR.exe -modeVAEGen -dataDir <dir> -resolutions 1920 1080 640 360 64 64 -numImages 10 -seed 1
set "PYTHONPATH=C:/Users/sambu/Documents/Repositories/CodeBases/SWiMM_DEEPeR;%PYTHONPATH%"
source ../../.venv/Scripts/activate
cd ../../benchmarking/scripts
python image_similarity_resizing.py --dirs_orig <dir1>,<dir2> --dirs_scaled <dir3>,<dir4> --dir_output <dir5>
python image_similarity_cross_resolution.py --dirs <dir6>,<dir7>,<dir8> --dir_output <dir9>
exit