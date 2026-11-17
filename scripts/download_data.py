"""Download the published UniDL4BioPep benchmark splits (no fabricated labels)."""
from pathlib import Path
import hashlib
import json
import urllib.parse
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "activity_train.xlsx": "5. Antimicrobial activity/AMP_train.xlsx",
    "activity_test.xlsx": "5. Antimicrobial activity/AMP_test.xlsx",
    "toxicity_train.xlsx": "17. Toxicity 2021 Dataset/train_3284.xlsx",
    "toxicity_test.xlsx": "17. Toxicity 2021 Dataset/test_580.xlsx",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', help='Explicit repository commit; otherwise reuse the existing source manifest.')
    args = parser.parse_args()
    target = ROOT / "Data" / "external"
    target.mkdir(parents=True, exist_ok=True)
    existing = target / 'sources.json'
    revision = args.revision or (json.loads(existing.read_text())['revision'] if existing.exists() else '23d025f3cd326556a7bb53f8e90fe60124345669')
    manifest = {"repository": "https://github.com/dzjxzyd/UniDL4BioPep", "revision": revision, "files": {}}
    for name, remote in FILES.items():
        url = f"https://raw.githubusercontent.com/dzjxzyd/UniDL4BioPep/{revision}/" + urllib.parse.quote(remote)
        data = urllib.request.urlopen(url, timeout=60).read()
        (target / name).write_bytes(data)
        manifest["files"][name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
        print(name, len(data), flush=True)
    (target / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
