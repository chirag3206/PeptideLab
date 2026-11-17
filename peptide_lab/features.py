"""Small, interpretable features shared by training and inference."""
from collections import Counter
import math
import numpy as np
from .config import AA, TOKEN, MAX_LENGTH

HYDRO = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -.8, -.7, 4.2, -.9, -1.3]))

def validate(sequence, min_length=5, max_length=MAX_LENGTH):
    sequence = "".join(str(sequence).split()).upper()
    if not min_length <= len(sequence) <= max_length:
        raise ValueError(f"Sequence length must be {min_length}–{max_length} amino acids.")
    invalid = set(sequence) - set(AA)
    if invalid:
        raise ValueError("Unsupported letters: " + ", ".join(sorted(invalid)))
    return sequence

def properties(sequence):
    c, n = Counter(sequence), len(sequence)
    return {
        "length": n,
        "charge_proxy": c['K'] + c['R'] - c['D'] - c['E'],
        "hydrophobic_fraction": sum(c[a] for a in "AILMFWVY") / n,
        "mean_hydropathy": sum(HYDRO[a] for a in sequence) / n,
        "entropy": -sum((v/n)*math.log2(v/n) for v in c.values()),
    }

FEATURE_NAMES = ["fraction_" + a for a in AA] + ["length", "charge_proxy", "hydrophobic_fraction", "mean_hydropathy", "entropy"]

def features(sequences):
    rows = []
    for s in sequences:
        c = Counter(s)
        rows.append([c[a] / len(s) for a in AA] + list(properties(s).values()))
    return np.asarray(rows, dtype=np.float32)

def encode(sequences, max_length=MAX_LENGTH):
    x = np.zeros((len(sequences), max_length), dtype=np.int64)
    for i, s in enumerate(sequences):
        x[i, :len(s)] = [TOKEN[a] for a in s]
    return x

def parse_sequences(text):
    """Accept one peptide per line or multiline FASTA; reject instead of silently changing residues."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Enter at least one peptide sequence.")
    if lines[0].startswith(">"):
        sequences, current = [], []
        for line in lines:
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line)
        if current:
            sequences.append("".join(current))
    else:
        sequences = lines
    if not sequences:
        raise ValueError("FASTA contains no sequence.")
    if len(sequences) > 500:
        raise ValueError("Use at most 500 sequences per batch.")
    return list(dict.fromkeys(validate(s) for s in sequences))
