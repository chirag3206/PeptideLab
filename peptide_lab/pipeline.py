"""Load saved models, screen sequences, and rank VAE-generated candidates."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
from .config import DATA, MODELS, RESULTS
from .features import validate, features, encode, properties
from .networks import PeptideCNN, PeptideVAE
from .training import cnn_predict

class ScreeningPipeline:
    def __init__(self):
        torch.set_num_threads(4)
        self.classifiers = {}
        for task in ('activity', 'toxicity'):
            metadata = joblib.load(MODELS / f'{task}_baseline.joblib')
            cnn = PeptideCNN()
            cnn.load_state_dict(torch.load(MODELS / f'{task}_cnn.pt', map_location='cpu', weights_only=True))
            cnn.eval()
            self.classifiers[task] = (metadata, cnn)
        self.vae = PeptideVAE()
        self.vae.load_state_dict(torch.load(MODELS / 'vae.pt', map_location='cpu', weights_only=True))
        self.vae.eval()
        self.known = set(pd.read_csv(DATA / 'processed' / 'grampa_clean_unique_sequences.csv').sequence)
        self.classifier_train = set()
        for task in ('activity', 'toxicity'):
            self.classifier_train.update(pd.read_csv(DATA / 'final' / f'{task}_train.csv').sequence)

    def predict(self, sequences):
        sequences = [validate(s) for s in sequences]
        if not sequences:
            raise ValueError('No sequences to screen.')
        x, tokens = features(sequences), encode(sequences)
        result = pd.DataFrame({'sequence': sequences})
        for task, (metadata, cnn) in self.classifiers.items():
            rf = metadata['forest'].predict_proba(x)[:, 1]
            cp = cnn_predict(cnn, tokens, metadata['scaler'].transform(x).astype(np.float32))
            predictions = {'cnn': cp, 'random_forest': rf, 'ensemble': (cp + rf)/2}
            result[task + '_score'] = predictions[metadata['selected']]
            result[task + '_model_disagreement'] = np.abs(cp - rf)
        # A transparent preference score, NOT the probability that a peptide is safe/effective.
        result['priority_score'] = 100 * result.activity_score * (1 - result.toxicity_score)
        result['activity_prediction'] = np.where(result.activity_score >= .5, 'AMP-like', 'Non-AMP-like')
        result['toxicity_prediction'] = np.where(result.toxicity_score >= .5, 'Toxic class', 'Non-toxic class')
        result['seen_in_grampa'] = result.sequence.isin(self.known)
        result['seen_in_classifier_training'] = result.sequence.isin(self.classifier_train)
        result = pd.concat([result, pd.DataFrame([properties(s) for s in sequences])], axis=1)
        return result.sort_values('priority_score', ascending=False).reset_index(drop=True)

    def generate(self, count=30, seed=42, temperature=.9, min_length=10, max_length=40):
        if not 1 <= count <= 200 or not 5 <= min_length <= max_length <= 60 or not .5 <= temperature <= 1.3:
            raise ValueError('Use 1–200 candidates, length 5–60, and temperature 0.5–1.3.')
        # Fixed bounded pool: report any shortage instead of pretending to fulfill it.
        sequences, ended = self.vae.sample(max(500, count * 15), seed, temperature, 60)
        accepted, seen = [], set()
        rejected = {'length': 0, 'truncated': 0, 'low_complexity': 0, 'known': 0, 'duplicate': 0}
        for sequence, natural_end in zip(sequences, ended):
            if not natural_end:
                rejected['truncated'] += 1
            elif not min_length <= len(sequence) <= max_length:
                rejected['length'] += 1
            elif properties(sequence)['entropy'] < 2.0 or max(sequence.count(a) for a in set(sequence))/len(sequence) > .5:
                rejected['low_complexity'] += 1
            elif sequence in self.known or sequence in self.classifier_train:
                rejected['known'] += 1
            elif sequence in seen:
                rejected['duplicate'] += 1
            else:
                seen.add(sequence)
                accepted.append(sequence)
        audit = {'sampled': len(sequences), 'accepted_before_limit': len(accepted), 'requested': count, 'returned': min(count, len(accepted)), 'rejected': rejected,
                 'seed': seed, 'temperature': temperature, 'length_range': [min_length, max_length],
                 'selection': 'First valid unique novel samples, then ranked by predictor scores. No laboratory validation.'}
        return (self.predict(accepted[:count]) if accepted else pd.DataFrame()), audit

def main():
    pipe = ScreeningPipeline()
    candidates, audit = pipe.generate(count=50)
    candidates.to_csv(RESULTS / 'ranked_candidates.csv', index=False)
    (RESULTS / 'generation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(candidates.head(5).to_string(index=False))

if __name__ == '__main__':
    main()
