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
    def __init__(self, engine='compact', ranking_rule='product', activity_weight=.5):
        if ranking_rule not in ['product', 'weighted_difference'] or not 0 <= activity_weight <= 1:
            raise ValueError('Invalid ranking rule or activity weight.')
        self.ranking_rule, self.activity_weight = ranking_rule, activity_weight
        if engine not in ['compact', 'esm2']:
            raise ValueError('Choose compact or esm2 prediction engine.')
        self.engine = engine
        self.esm = None
        if engine == 'esm2':
            from .esm_predictor import ESMPredictor
            self.esm = ESMPredictor()
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
        esm_scores = self.esm.predict(sequences) if self.esm is not None else None
        for task, (metadata, cnn) in self.classifiers.items():
            rf = metadata['forest'].predict_proba(x)[:, 1]
            cp = cnn_predict(cnn, tokens, metadata['scaler'].transform(x).astype(np.float32))
            predictions = {'cnn': cp, 'random_forest': rf, 'ensemble': (cp + rf)/2}
            baseline = predictions[metadata['selected']]
            result[task + '_score'] = esm_scores[task] if esm_scores is not None else baseline
            result[task + '_compact_score'] = baseline
            if esm_scores is not None:
                result[task + '_esm2_score'] = esm_scores[task]
            result[task + '_model_disagreement'] = np.abs(cp - rf)
        # A transparent preference score, NOT the probability that a peptide is safe/effective.
        result['priority_score'] = 100 * result.activity_score * (1 - result.toxicity_score)
        result['weighted_difference_score'] = self.activity_weight * result.activity_score - (1-self.activity_weight) * result.toxicity_score
        if self.ranking_rule == 'weighted_difference':
            result['priority_score'] = 100 * (result.weighted_difference_score + 1-self.activity_weight)
        result['ranking_rule'] = self.ranking_rule
        result['activity_weight'] = self.activity_weight
        result['prediction_engine'] = self.engine
        threshold_path = RESULTS / 'alignment/property_thresholds.json'
        if threshold_path.exists():
            thresholds = json.loads(threshold_path.read_text())[self.engine]
            result['activity_screen_threshold'] = thresholds['activity']['threshold']
            result['toxicity_screen_threshold'] = thresholds['toxicity']['threshold']
            result['passes_property_screen'] = (result.activity_score >= thresholds['activity']['threshold']) & (result.toxicity_score < thresholds['toxicity']['threshold'])
        result['activity_prediction'] = np.where(result.activity_score >= .5, 'AMP-like', 'Non-AMP-like')
        result['toxicity_prediction'] = np.where(result.toxicity_score >= .5, 'Toxic class', 'Non-toxic class')
        result['seen_in_grampa'] = result.sequence.isin(self.known)
        result['seen_in_classifier_training'] = result.sequence.isin(self.classifier_train)
        result = pd.concat([result, pd.DataFrame([properties(s) for s in sequences])], axis=1)
        result = result.sort_values(['priority_score', 'sequence'], ascending=[False, True], kind='stable').reset_index(drop=True)
        result.insert(0, 'rank', np.arange(1, len(result) + 1))
        return result

    def generate(self, count=500, seed=42, temperature=.9, min_length=10, max_length=40):
        if not 1 <= count <= 3000 or not 5 <= min_length <= max_length <= 60 or not .5 <= temperature <= 1.3:
            raise ValueError('Use 1-3000 raw samples, length 5-60, and temperature 0.5-1.3.')
        # Score every accepted sequence from the explicitly sized sampling pool.
        sequences, ended = self.vae.sample(count, seed, temperature, 60)
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
        audit = {'sampled': len(sequences), 'passed_filters': len(accepted), 'returned': len(accepted), 'rejected': rejected,
                 'seed': seed, 'temperature': temperature, 'length_range': [min_length, max_length],
                 'selection': 'All passed-filter sequences scored and ranked together. No laboratory validation.'}
        frame = self.predict(accepted) if accepted else pd.DataFrame()
        audit['prediction_engine'] = self.engine
        audit['ranking_rule'] = self.ranking_rule
        if 'passes_property_screen' in frame:
            audit['passed_property_screen'] = int(frame.passes_property_screen.sum())
        return frame, audit

def main():
    pipe = ScreeningPipeline()
    candidates, audit = pipe.generate(count=750)
    candidates.to_csv(RESULTS / 'ranked_candidates.csv', index=False)
    (RESULTS / 'generation_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(candidates.head(5).to_string(index=False))

if __name__ == '__main__':
    main()
