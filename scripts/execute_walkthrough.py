"""Execute our teaching notebook in-process and save genuine text/table/plot outputs.

No Jupyter server is required. This runner supports this project's plain Python
cells only; use Jupyter/VS Code for arbitrary notebooks and interactive editing.
"""
from pathlib import Path
import base64
import contextlib
import io
import json
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--notebook', default='Notebooks/03_Final_Project_Walkthrough.ipynb')
args = parser.parse_args()
path = ROOT / args.notebook
nb = json.loads(path.read_text(encoding='utf-8'))
scope = {'__name__': '__main__'}
counter = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    counter += 1
    outputs = []
    def display(value):
        data = {'text/plain': str(value)}
        if hasattr(value, 'to_html'):
            data['text/html'] = value.to_html()
        outputs.append({'output_type': 'display_data', 'data': data, 'metadata': {}})
    def show():
        for number in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(number).savefig(buf, format='png', bbox_inches='tight')
            outputs.append({'output_type': 'display_data', 'data': {'image/png': base64.b64encode(buf.getvalue()).decode()}, 'metadata': {}})
        plt.close('all')
    scope['display'] = display
    plt.show = show
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(''.join(cell['source']), str(path) + f':cell{counter}', 'exec'), scope)
    if stream.getvalue():
        outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
    cell['outputs'] = outputs
    cell['execution_count'] = counter
    print('Executed cell', counter, flush=True)
path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
print('Saved executed walkthrough:', path)
