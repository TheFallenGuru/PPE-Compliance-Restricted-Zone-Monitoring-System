@echo off
setlocal
title PPE Safety Monitor - Service Stopper
echo =======================================================
echo       PPE SAFETY MONITOR - STOPPING ALL SERVICES
echo =======================================================
echo.

echo Stopping Intrusion Watcher...
taskkill /FI "WINDOWTITLE eq PPE Intrusion Watcher*" /T /F >nul 2>&1

echo Stopping PPE Safety Monitor HMI...
taskkill /FI "WINDOWTITLE eq PPE Safety Monitor*" /T /F >nul 2>&1

echo.
echo All PPE Monitoring services have been stopped.
timeout /t 2 >nul
