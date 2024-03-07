@echo off
setlocal enabledelayedexpansion
set "FILE_PATH=tensorboard_dir.txt"
set "SEARCH_STRING=Tensorboard event file:"

call :processFile
goto :eof

:processFile
for /F "tokens=4 delims= " %%a in ('findstr /I /R /C:"%SEARCH_STRING%" %FILE_PATH% 2^>nul') do (
    set "s=%%a"
    set "SearchFound=1"
)
if not defined SearchFound (
    echo Searching for logging file...
    timeout /nobreak /t 1 > nul
    goto :processFile
) else (
    echo Logging file found: !s!
    del "%FILE_PATH%"
    goto :found
)

:found
@REM for %%I in ("%s%") do set "dir=%%~dpI"
@REM if "%dir:~-1%"=="\" set "dir=%dir:~0,-1%"
call ..\.venv\Scripts\activate
echo Starting TensorBoard...
start /B tensorboard --host localhost --port 6006 --logdir "%s%" --reload_interval=10
timeout /nobreak /t 5 > nul
for /f "tokens=5" %%a in ('netstat -ano ^| find "6006" ^| find "LISTENING"') do (
    set "tensorboardPID=%%a"
    echo "TensorBoardPID: %%tensorboardPID"
)
if not defined tensorboardPID (
    echo Failed to find TensorBoard process.
    exit 1
)
echo TensorBoard started with PID: !tensorboardPID!
start "" "http://localhost:6006/"
pause
taskkill /F /PID %tensorboardPID%
goto :eof

:eof
endlocal
exit 0