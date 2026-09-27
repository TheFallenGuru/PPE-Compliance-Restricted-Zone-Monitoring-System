@echo off
setlocal enabledelayedexpansion
title PPE Compliance Monitor - Tkinter GUI Launcher
cd /d "%~dp0"

echo =======================================================
echo       PPE COMPLIANCE MONITORING SYSTEM (DESKTOP)
echo             Tkinter Desktop GUI Launcher
echo =======================================================
echo.

set "PYTHON_EXE="
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

set "PYTHONW_EXE="
if exist "%~dp0venv\Scripts\pythonw.exe" (
    set "PYTHONW_EXE=%~dp0venv\Scripts\pythonw.exe"
) else if exist "%~dp0.venv\Scripts\pythonw.exe" (
    set "PYTHONW_EXE=%~dp0.venv\Scripts\pythonw.exe"
) else (
    where pythonw.exe >nul 2>&1
    if not errorlevel 1 (
        set "PYTHONW_EXE=pythonw"
    )
)

:: Start the real-time intrusion log watcher silently in the background (no console window)
if defined PYTHONW_EXE (
    start "" "%PYTHONW_EXE%" "%~dp0scripts\intrusion_log_watcher.py" --watch-dir "%~dp0output"
) else (
    start "PPE Intrusion Watcher" /min "%PYTHON_EXE%" "%~dp0scripts\intrusion_log_watcher.py" --watch-dir "%~dp0output"
)

:: Start the main PPE compliance Tkinter GUI application
"%PYTHON_EXE%" "%~dp0app.py"

:: Cleanly close the background intrusion watcher when the main application exits
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object ProcessName -like '*python*' | Where-Object CommandLine -like '*intrusion_log_watcher.py*' | Stop-Process -Force" >nul 2>&1
wmic process where "caption='pythonw.exe' and commandline like '%%intrusion_log_watcher.py%%'" call terminate >nul 2>&1
taskkill /FI "WINDOWTITLE eq PPE Intrusion Watcher*" /T /F >nul 2>&1
