CALL .\..\.venv\Scripts\activate.bat
cd ..\Builds\Windows
START SWiMM_DEEPeR.exe mode_sample_gen num_images 360 data_dir D:\sampling\ resolutions 640 360 854 480 960 540 1280 720 1600 900 1920 1080 2560 1440 3200 1800 3840 2160 5120 2880 7860 4320
EXIT