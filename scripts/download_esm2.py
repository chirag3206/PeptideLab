"""Fetch the official small ESM-2 checkpoint with pinned revision and hashes."""
from pathlib import Path
import urllib.request
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def main():
    folder = ROOT / 'Models' / 'esm2_backbone'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'source.json'
    info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/facebook/esm2_t6_8M_UR50D', timeout=60))
    revision = json.loads(manifest_path.read_text())['revision'] if manifest_path.exists() else info['sha']
    available = {x['rfilename'] for x in info['siblings']}
    manifest = {'model': 'facebook/esm2_t6_8M_UR50D', 'revision': revision, 'license': 'MIT (see official model card)', 'files': {}}
    for name in ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'model.safetensors', 'README.md']:
        if name not in available:
            if name in ['config.json', 'vocab.txt', 'model.safetensors']:
                raise RuntimeError('Required model file missing: ' + name)
            continue
        url = f'https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/{revision}/{name}'
        path = folder / name
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=180).read())
        manifest['files'][name] = {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, path.stat().st_size, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
