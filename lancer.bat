@echo off
setlocal
cd /d "%~dp0"
title Tap Software

if not exist venv\Scripts\python.exe goto notinstalled

call venv\Scripts\activate.bat

REM open the browser only when http://localhost:5000 answers (checks every second, opens anyway after 60 s)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "$t=0; while ($t -lt 60) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri 'http://localhost:5000/' | Out-Null; break } catch { Start-Sleep -Seconds 1; $t++ } }; Start-Process 'http://localhost:5000'"

echo Tap Software is starting. The browser opens as soon as the app is ready.
echo Close this window to stop it.
echo Participant page : http://localhost:5000
echo Examiner page    : http://localhost:5000/examiner
echo.
python app.py
pause
exit /b 0

:notinstalled
echo The app is not installed yet. Run "installer.bat" first.
pause
exit /b 1
