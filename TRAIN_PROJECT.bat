@echo off
cd /d "%~dp0"
python -m peptide_lab.training
if errorlevel 1 goto failed
python -m peptide_lab.screening_thresholds
if errorlevel 1 goto failed
python -m peptide_lab.pipeline
if errorlevel 1 goto failed
python scripts/build_report.py
if errorlevel 1 goto failed
python scripts/execute_walkthrough.py
if errorlevel 1 goto failed
echo Baseline training completed. Run FINALIZE_PROJECT.bat to refresh the complete report and evidence.
echo Run python -m peptide_lab.esm_predictor separately to retrain the ESM heads.
pause
exit /b 0
:failed
echo A step failed. Read the error above before rerunning.
pause
exit /b 1
