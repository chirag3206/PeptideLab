"""Reconstruct GRAMPA preprocessing from raw data without overwriting history."""
import hashlib
import json
import pandas as pd
from .config import DATA, RESULTS, AA, prepare_directories

def prepare():
    prepare_directories()
    raw_path = DATA / 'raw/grampa.csv'
    raw = pd.read_csv(raw_path)
    seq = raw.sequence.astype('string').str.strip().str.upper()
    valid = seq.str.fullmatch(f'[{AA}]+').fillna(False)
    clean = pd.DataFrame({'sequence': seq[valid].drop_duplicates()}).reset_index(drop=True)
    clean['sequence_length'] = clean.sequence.str.len()
    path = DATA / 'final/grampa_canonical.csv'
    clean.to_csv(path, index=False)
    counts = {}
    for limit in [40, 60]:
        filtered = clean[clean.sequence_length.between(5, limit)]
        filtered.to_csv(DATA / 'final' / f'grampa_5_{limit}.csv', index=False)
        counts[f'5_{limit}'] = len(filtered)
    historical = pd.read_csv(DATA / 'processed/grampa_clean_unique_sequences.csv')
    audit = {'raw_records': len(raw), 'missing_sequences': int(seq.isna().sum()),
             'invalid_sequence_records': int((~valid).sum()), 'duplicate_valid_records_removed': int(valid.sum()-len(clean)),
             'canonical_unique_sequences': len(clean), 'length_counts': counts,
             'matches_historical_clean_set': set(clean.sequence)==set(historical.sequence),
             'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
             'clean_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'policy': 'Strip and uppercase sequence; retain 20 canonical residues; deduplicate by sequence preserving first appearance; apply explicit length bounds. No species or modification filter. Sequence-only models do not model chemical modifications or assay conditions.'}
    (RESULTS / 'preprocessing_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit

if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
