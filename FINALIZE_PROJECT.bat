@echo off
cd /d "%~dp0"
python -m peptide_lab.screening_thresholds
if errorlevel 1 goto failed
python -m peptide_lab.pipeline
if errorlevel 1 goto failed
python -m peptide_lab.alignment_checks
if errorlevel 1 goto failed
python scripts/build_report.py
if errorlevel 1 goto failed
python scripts/build_extended_report.py
if errorlevel 1 goto failed
python scripts/build_alignment_report.py
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py --notebook Notebooks/04_Dataset_Prediction_and_Research.ipynb
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py --notebook Notebooks/05_Final_Alignment_and_ESM2.ipynb
if errorlevel 1 goto failed
python scripts/build_master_guide.py
if errorlevel 1 goto failed
python scripts/verify_project.py
if errorlevel 1 goto failed
echo Final evidence, reports and executed notebooks are updated. No production model was retrained.
pause
exit /b 0
:failed
echo A step failed. Read the error above.
pause
exit /b 1
