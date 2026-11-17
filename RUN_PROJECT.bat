@echo off
cd /d "%~dp0"
echo Starting PeptideLab at http://127.0.0.1:8501
python -m streamlit run app.py --server.headless false
pause
