@echo off
cd /d "%~dp0"
python -m peptide_lab.evaluation
if errorlevel 1 goto failed
python -m peptide_lab.toy_study
if errorlevel 1 goto failed
python scripts/build_extended_report.py
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py --notebook Notebooks/04_Dataset_Prediction_and_Research.ipynb
if errorlevel 1 goto failed
python scripts/verify_project.py
if errorlevel 1 goto failed
echo Extended evaluation completed. Deployed model weights were not replaced.
pause
exit /b 0
:failed
echo A step failed. Read the error above.
pause
exit /b 1
