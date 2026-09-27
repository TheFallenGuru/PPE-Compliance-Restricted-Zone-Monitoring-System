@echo off
setlocal
title PPE Compliance Monitor - Test Suite
cd /d "%~dp0"

echo =======================================================
echo     PPE COMPLIANCE MONITOR - REGRESSION TEST SUITE
echo =======================================================
echo.

python -m unittest discover -s tests -p "test_*.py" -v

echo.
pause
