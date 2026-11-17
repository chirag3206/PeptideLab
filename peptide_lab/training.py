"""Reproducible training; model selection uses validation data, never test scores."""
import argparse
import copy
import json
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import DATA, MODELS, RESULTS, AA, START, END, SEED, prepare_directories
from .features import encode, features, FEATURE_NAMES
from .networks import PeptideCNN, PeptideVAE

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

def metrics(y, p):
    prediction = np.asarray(p) >= .5
    return {"accuracy": accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0),
            "mcc": matthews_corrcoef(y, prediction), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p), "confusion_matrix": confusion_matrix(y, prediction).tolist()}

def load_splits(task):
    frames = {part: pd.read_excel(DATA / 'external' / f'{task}_{part}.xlsx') for part in ('train', 'test')}
    audit = {"raw_rows": {k: len(v) for k, v in frames.items()}}
    for key, df in frames.items():
        df['sequence'] = df.sequence.astype(str).str.strip().str.upper()
        valid = df.sequence.str.fullmatch(f'[{AA}]{{5,200}}') & df.label.isin([0, 1])
        frames[key] = df[valid].copy()
        audit[key + '_invalid_or_out_of_length'] = int((~valid).sum())
    all_rows = pd.concat(list(frames.values()))
    conflicting = set(all_rows.groupby('sequence').label.nunique().loc[lambda s: s > 1].index)
    audit['conflicting_sequences_removed'] = len(conflicting)
    for key, df in frames.items():
        frames[key] = df[~df.sequence.isin(conflicting)].drop_duplicates('sequence').reset_index(drop=True)
    overlap = set(frames['train'].sequence) & set(frames['test'].sequence)
    audit['train_test_overlap_removed_from_train'] = len(overlap)
    pool = frames['train'][~frames['train'].sequence.isin(overlap)]
    train, val = train_test_split(pool, test_size=.2, stratify=pool.label, random_state=SEED)
    frames = {'train': train.reset_index(drop=True), 'validation': val.reset_index(drop=True), 'test': frames['test']}
    audit['splits'] = {key: {'rows': len(df), 'positive': int(df.label.sum()), 'negative': int((df.label == 0).sum())} for key, df in frames.items()}
    audit['policy'] = 'Published test split retained; conflicting labels removed globally; exact duplicates and cross-split overlaps removed; 20% of training reserved for validation. Not a homology-clustered split.'
    for key, df in frames.items():
        df.to_csv(DATA / 'final' / f'{task}_{key}.csv', index=False)
    write_json(RESULTS / f'{task}_data_audit.json', audit)
    return frames

@torch.no_grad()
def cnn_predict(model, tokens, descriptors):
    model.eval()
    out = []
    for start in range(0, len(tokens), 256):
        out.append(torch.sigmoid(model(torch.as_tensor(tokens[start:start+256]), torch.as_tensor(descriptors[start:start+256], dtype=torch.float32))).numpy())
    return np.concatenate(out)

def train_classifier(task, epochs=18):
    seed_everything()
    start = time.time()
    frames = load_splits(task)
    xs = {k: features(d.sequence.tolist()) for k, d in frames.items()}
    tokens = {k: encode(d.sequence.tolist()) for k, d in frames.items()}
    ys = {k: d.label.to_numpy(dtype=np.float32) for k, d in frames.items()}
    scaler = StandardScaler().fit(xs['train'])
    scaled = {k: scaler.transform(x).astype(np.float32) for k, x in xs.items()}
    forest = RandomForestClassifier(n_estimators=240, min_samples_leaf=2, max_features=.7, class_weight='balanced', n_jobs=4, random_state=SEED)
    forest.fit(xs['train'], ys['train'])
    model = PeptideCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    weight = torch.tensor(float((ys['train'] == 0).sum() / (ys['train'] == 1).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    loader = DataLoader(TensorDataset(torch.as_tensor(tokens['train']), torch.as_tensor(scaled['train']), torch.as_tensor(ys['train'])), batch_size=128, shuffle=True)
    best_auc, best, best_epoch, stale, history = -1, None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for x, desc, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, desc), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        vp = cnn_predict(model, tokens['validation'], scaled['validation'])
        auc = roc_auc_score(ys['validation'], vp)
        history.append({'epoch': epoch, 'training_loss': total / len(ys['train']), 'validation_auc': auc})
        print(f'{task} epoch {epoch}: loss={history[-1]["training_loss"]:.4f}, validation AUC={auc:.4f}', flush=True)
        if auc > best_auc + .0001:
            best_auc, best, best_epoch, stale = auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best)
    val_cnn = cnn_predict(model, tokens['validation'], scaled['validation'])
    val_rf = forest.predict_proba(xs['validation'])[:, 1]
    validation_auc = {name: roc_auc_score(ys['validation'], p) for name, p in {'cnn': val_cnn, 'random_forest': val_rf, 'ensemble': (val_cnn + val_rf)/2}.items()}
    selected = max(validation_auc, key=validation_auc.get)
    torch.save(model.state_dict(), MODELS / f'{task}_cnn.pt')
    joblib.dump({'scaler': scaler, 'forest': forest, 'selected': selected}, MODELS / f'{task}_baseline.joblib')
    # Only after all selection is finished do we evaluate the held-out test data.
    test_cnn = cnn_predict(model, tokens['test'], scaled['test'])
    test_rf = forest.predict_proba(xs['test'])[:, 1]
    predictions = {'cnn': test_cnn, 'random_forest': test_rf, 'ensemble': (test_cnn + test_rf)/2}
    test_metrics = {name: metrics(ys['test'], p) for name, p in predictions.items()}
    df = frames['test'].copy()
    for name, p in predictions.items():
        df[name + '_score'] = p
    df.to_csv(RESULTS / f'{task}_test_predictions.csv', index=False)
    pd.DataFrame(history).to_csv(RESULTS / f'{task}_history.csv', index=False)
    pd.DataFrame({'feature': FEATURE_NAMES, 'importance': forest.feature_importances_}).sort_values('importance', ascending=False).to_csv(RESULTS / f'{task}_feature_importance.csv', index=False)
    report = {'task': task, 'positive_label': 'antimicrobial' if task == 'activity' else 'toxic', 'selected_model': selected, 'best_cnn_epoch': best_epoch,
              'validation_auc': validation_auc, 'test': test_metrics, 'threshold': .5, 'seconds': time.time()-start,
              'limitations': 'Sequence-level benchmark, not homology-disjoint. Scores are not experimentally validated or calibrated probabilities. Activity is binary AMP classification, not MIC or species-specific potency.'}
    write_json(RESULTS / f'{task}_metrics.json', report)
    print(task, 'selected', selected, test_metrics[selected], flush=True)

def vae_arrays(sequences, length=61):
    inputs = np.zeros((len(sequences), length), dtype=np.int64)
    targets = np.zeros_like(inputs)
    from .config import TOKEN
    for i, s in enumerate(sequences):
        ids = [TOKEN[a] for a in s]
        inputs[i, :len(s)+1] = [START] + ids
        targets[i, :len(s)+1] = ids + [END]
    return torch.as_tensor(inputs), torch.as_tensor(targets)

def train_vae(epochs=30):
    seed_everything()
    from .preprocessing import prepare
    prepare()
    source = pd.read_csv(DATA / 'final' / 'grampa_canonical.csv')
    sequences = source.sequence.drop_duplicates()
    sequences = sequences[sequences.str.fullmatch(f'[{AA}]{{5,60}}')].tolist()
    train, held = train_test_split(sequences, test_size=.2, random_state=SEED)
    val, test = train_test_split(held, test_size=.5, random_state=SEED)
    splits = {'train': train, 'validation': val, 'test': test}
    for key, seqs in splits.items():
        pd.DataFrame({'sequence': seqs}).to_csv(DATA / 'final' / f'vae_{key}.csv', index=False)
    arrays = {k: vae_arrays(v) for k, v in splits.items()}
    loaders = {k: DataLoader(TensorDataset(*v), batch_size=128, shuffle=k=='train') for k, v in arrays.items()}
    model = PeptideVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    best, best_loss, stale, history = None, float('inf'), 0, []
    for epoch in range(1, epochs + 1):
        # Gradually introduce the KL regularizer; keep a small fixed weight after warmup.
        beta = .02 * min(1., epoch / 10)
        stats = {}
        for split in ['train', 'validation']:
            model.train(split == 'train')
            total_recon, total_kl, n_tokens, n_rows = 0., 0., 0, 0
            with torch.set_grad_enabled(split == 'train'):
                for x, y in loaders[split]:
                    logits, mu, logvar = model(x)
                    recon_sum = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0, reduction='sum')
                    count = int(y.ne(0).sum())
                    kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
                    loss = recon_sum / count + beta * kl
                    if split == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                    total_recon += recon_sum.item()
                    total_kl += kl.item() * len(x)
                    n_tokens += count
                    n_rows += len(x)
            stats[split + '_reconstruction'] = total_recon / n_tokens
            stats[split + '_kl'] = total_kl / n_rows
        # Compare a fixed objective across epochs, independent of warmup beta.
        objective = stats['validation_reconstruction'] + .02 * stats['validation_kl']
        history.append({'epoch': epoch, 'beta': beta, **stats})
        print(f'VAE epoch {epoch}: train CE={stats["train_reconstruction"]:.4f}, val CE={stats["validation_reconstruction"]:.4f}, KL={stats["validation_kl"]:.4f}', flush=True)
        if objective < best_loss:
            best_loss, best, stale = objective, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best)
    model.eval()
    torch.save(model.state_dict(), MODELS / 'vae.pt')
    pd.DataFrame(history).to_csv(RESULTS / 'vae_history.csv', index=False)
    correct, total, loss_sum = 0, 0, 0.
    with torch.no_grad():
        for x, y in loaders['test']:
            logits, _, _ = model(x)
            correct += int(((logits.argmax(-1) == y) & y.ne(0)).sum())
            total += int(y.ne(0).sum())
            loss_sum += nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=0, reduction='sum').item()
        mu, _ = model.encode(arrays['test'][0])
        pd.DataFrame(mu.numpy()).to_csv(RESULTS / 'vae_test_latent.csv', index=False)
    generated, ended = model.sample(500, seed=SEED)
    known = set(source.sequence)
    unique = set(generated)
    pd.DataFrame({'sequence': generated, 'ended_naturally': ended, 'seen_in_grampa': [s in known for s in generated]}).to_csv(RESULTS / 'vae_raw_generation.csv', index=False)
    from .features import properties
    report = {'architecture': 'LSTM VAE: embedding 32, hidden 96, latent 24; word dropout 0.2; KL warmup to 0.02',
              'splits': {k: len(v) for k, v in splits.items()}, 'length_range': [5, 60],
              'test_teacher_forced_token_accuracy': correct/total, 'test_teacher_forced_cross_entropy': loss_sum/total,
              'generated': len(generated), 'unique': len(unique), 'uniqueness': len(unique)/len(generated),
              'novel_unique_vs_grampa': sum(s not in known for s in unique)/len(unique),
              'natural_end_rate': sum(ended)/len(ended), 'mean_entropy': float(np.mean([properties(s)['entropy'] for s in generated])),
              'notes': 'Sampling forbids special tokens and END before 5 residues, and caps length at 60. Alphabet/length validity is enforced by decoding, not evidence of biological validity. Novelty means exact non-match to GRAMPA only. Test reconstruction uses true previous tokens and is easier than free generation.'}
    write_json(RESULTS / 'vae_metrics.json', report)
    print('VAE evaluation', report, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['all', 'activity', 'toxicity', 'vae'], default='all')
    parser.add_argument('--classifier-epochs', type=int, default=18)
    parser.add_argument('--vae-epochs', type=int, default=30)
    args = parser.parse_args()
    prepare_directories()
    # Keep the previous demonstration artifacts before any intentional retraining.
    if any(MODELS.iterdir()):
        from datetime import datetime, timezone
        import shutil
        from .config import ROOT
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive = ROOT / 'Results' / 'archives' / stamp
        shutil.copytree(MODELS, archive / 'models')
        shutil.copytree(RESULTS, archive / 'results')
        shutil.copytree(DATA / 'final', archive / 'data_splits')
        write_json(archive / 'next_training_arguments.json', vars(args))
        print('Previous artifacts preserved in', archive, flush=True)
    for task in ('activity', 'toxicity'):
        if args.component in ('all', task):
            train_classifier(task, args.classifier_epochs)
    if args.component in ('all', 'vae'):
        train_vae(args.vae_epochs)

if __name__ == '__main__':
    main()
