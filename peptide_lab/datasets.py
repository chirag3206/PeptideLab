"""Dataset browsing with explicit labels and per-model split membership."""
import pandas as pd
from .config import DATA, AA

SOURCES = ['GRAMPA', 'Activity benchmark', 'Toxicity benchmark']

def catalog():
    """Join on sequence, never by row position. Missing labels remain missing."""
    grampa = pd.read_csv(DATA / 'processed/grampa_clean_unique_sequences.csv')[['sequence']]
    grampa = grampa[grampa.sequence.str.fullmatch(f'[{AA}]{{5,200}}')].drop_duplicates()
    grampa['in_grampa'] = True
    combined = grampa
    for task in ['activity', 'toxicity']:
        frames = []
        for split in ['train', 'validation', 'test']:
            frame = pd.read_csv(DATA / 'final' / f'{task}_{split}.csv')[['sequence', 'label']]
            frame = frame.rename(columns={'label': task + '_label'})
            frame[task + '_split'] = split
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        if frame.sequence.duplicated().any():
            raise ValueError(f'Duplicate sequences across {task} splits; rebuild clean splits before browsing.')
        combined = combined.merge(frame, on='sequence', how='outer', validate='one_to_one')
    combined['in_grampa'] = combined.in_grampa.eq(True)
    for task in ['activity', 'toxicity']:
        combined[task + '_label'] = combined[task + '_label'].astype('Int64')
        combined[task + '_split'] = combined[task + '_split'].fillna('not in benchmark')
    combined['length'] = combined.sequence.str.len()
    return combined.sort_values('sequence').reset_index(drop=True)

def browse(frame, source='GRAMPA', split='All', label='All', min_length=5, max_length=200, query=''):
    out = frame.copy()
    task = {'Activity benchmark': 'activity', 'Toxicity benchmark': 'toxicity'}.get(source)
    if source == 'GRAMPA':
        out = out[out.in_grampa]
    elif task:
        out = out[out[task + '_label'].notna()]
        if split != 'All':
            out = out[out[task + '_split'].eq(split)]
        if label != 'All':
            out = out[out[task + '_label'].eq(int(label))]
    else:
        raise ValueError('Unknown dataset source.')
    out = out[out.length.between(min_length, max_length)]
    if query.strip():
        out = out[out.sequence.str.contains(query.strip().upper(), regex=False)]
    return out.reset_index(drop=True)

def attach_labels(predictions, selected):
    labels = selected.drop(columns=['length'], errors='ignore')
    out = predictions.merge(labels, on='sequence', how='left', validate='one_to_one')
    for task in ['activity', 'toxicity']:
        column = task + '_label'
        if column not in out:
            continue
        observed = out[column].notna()
        out[task + '_matches_label'] = pd.Series(pd.NA, index=out.index, dtype='boolean')
        out.loc[observed, task + '_matches_label'] = (out.loc[observed, task + '_score'].ge(.5).astype(int).to_numpy() == out.loc[observed, column].astype(int).to_numpy())
    return out

