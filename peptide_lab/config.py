from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
MODELS = ROOT / "Models" / "final"
RESULTS = ROOT / "Results" / "final"
AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, START, END = 0, 21, 22
TOKEN = {a: i + 1 for i, a in enumerate(AA)}
SEED = 42
MAX_LENGTH = 200

def prepare_directories():
    for p in (MODELS, RESULTS, DATA / "final"):
        p.mkdir(parents=True, exist_ok=True)
