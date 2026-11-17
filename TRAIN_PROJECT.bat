@echo off
cd /d "%~dp0"
python -m peptide_lab.training
if errorlevel 1 goto failed
python -m peptide_lab.pipeline
if errorlevel 1 goto failed
python scripts/build_report.py
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py
if errorlevel 1 goto failed
echo Training and evaluation completed.
pause
exit /b 0
:failed
echo A step failed. Read the error above before rerunning.
pause
exit /b 1
