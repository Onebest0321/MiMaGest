@echo off
setlocal
cd /d "%~dp0"
title Tap Software - Installation

REM ---------------------------------------------------------------
REM  Accepted Python versions, in order of preference.
REM  Put the version the app was tested with first.
REM ---------------------------------------------------------------
set PYVER=
for %%V in (3.12) do (
  if not defined PYVER (
    py -%%V --version >nul 2>&1 && set PYVER=%%V
  )
)

echo ============================================
echo   Tap Software - Installation
echo ============================================
echo.

REM --- 1. Check that a supported Python was found ----------------
if not defined PYVER goto nopython
echo Using Python %PYVER%

REM --- 2. Create the private environment -------------------------
if exist venv\Scripts\python.exe goto venvok
echo Creating the environment...
py -%PYVER% -m venv venv
if errorlevel 1 goto fail
:venvok

REM --- 3. Install the exact library versions ---------------------
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto fail

if exist wheels\ goto offline
echo Installing libraries (internet required, this can take several minutes)...
pip install -r requirements.txt
if errorlevel 1 goto fail
goto done

:offline
echo Installing libraries from the "wheels" folder (no internet needed)...
pip install --no-index --find-links wheels -r requirements.txt
if errorlevel 1 goto fail

:done
echo.
echo ============================================
echo   Installation finished.
echo   Start the app with "lancer.bat".
echo ============================================
pause
exit /b 0

:nopython
echo No supported Python was found (3.12 or 3.11).
echo.
echo 1. Download Python 3.12 from https://www.python.org/downloads/windows/
echo 2. During installation, tick "Add Python to PATH".
echo 3. Run this file again.
echo.
pause
exit /b 1

:fail
echo.
echo The installation failed. Read the messages above,
echo then see the "Troubleshooting" part of README.txt.
echo.
pause
exit /b 1
