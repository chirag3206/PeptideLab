"""Run tests and retain their actual output and exit status on Windows/Linux."""
from pathlib import Path
import subprocess
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
folder=ROOT/'Results/final/extended'
folder.mkdir(parents=True,exist_ok=True)
output=result.stdout+result.stderr
(folder/'verification_tests.txt').write_text(output,encoding='utf-8')
(folder/'verification_status.json').write_text(json.dumps({'command':'python -B -m unittest discover -s tests -v','exit_code':result.returncode},indent=2))
sys.stdout.reconfigure(encoding='utf-8')
print(output)
raise SystemExit(result.returncode)
