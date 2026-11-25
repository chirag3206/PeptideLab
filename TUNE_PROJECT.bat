@echo off
cd /d "%~dp0"
python -m peptide_lab.tuning
if errorlevel 1 goto failed
echo Tuning completed. Open Model results, choose a model tab, then Hyperparameter tuning.
pause
exit /b 0
:failed
echo Tuning failed. Read the error above. Completed trials are retained for resume.
pause
exit /b 1
