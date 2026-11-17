"""Choose property-screen cutoffs using validation labels only, never test labels."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from .config import DATA, RESULTS
from .pipeline import ScreeningPipeline

def main():
    output = {}
    for engine in ['compact', 'esm2']:
        pipe = ScreeningPipeline(engine)
        output[engine] = {}
        for task in ['activity', 'toxicity']:
            path = DATA/'final'/f'{task}_validation.csv'
            frame = pd.read_csv(path)
            predicted = pipe.predict(frame.sequence.tolist()).set_index('sequence')
            score = predicted.loc[frame.sequence, task+'_score'].to_numpy()
            y = frame.label.to_numpy()
            thresholds = np.linspace(.01, .99, 99)
            if task == 'activity':
                threshold = max(thresholds, key=lambda t: (f1_score(y, score >= t), t))
                rule = 'Maximize validation F1 over 0.01..0.99; ties use higher cutoff.'
            else:
                eligible = [t for t in thresholds if recall_score(y, score >= t) >= .95]
                threshold = max(eligible) if eligible else 0.0
                rule = 'Highest grid cutoff with validation toxic-class recall >= 0.95; fallback zero.'
            output[engine][task] = {'threshold':float(threshold), 'rule':rule,
                'validation_rows':len(y), 'validation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'recall':recall_score(y, score>=threshold), 'precision':precision_score(y, score>=threshold,zero_division=0),
                'f1':f1_score(y, score>=threshold), 'protocol':'Follow-up validation-only screening rule; benchmark classification metrics retain threshold 0.5. Not a safety guarantee.'}
            frame[engine+'_score'] = score
            frame.to_csv(RESULTS/'alignment'/f'{engine}_{task}_validation_predictions.csv',index=False)
    (RESULTS/'alignment/property_thresholds.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    main()
