@echo off
cd /d "%~dp0"
echo Starting PeptideLab at http://127.0.0.1:8501
setlocal
set "PEPTIDELAB_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PEPTIDELAB_PYTHON=.venv\Scripts\python.exe"
"%PEPTIDELAB_PYTHON%" -m streamlit run app.py --server.headless false
pause
