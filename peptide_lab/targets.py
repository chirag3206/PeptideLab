"""Compare models of measured MIC using GRAMPA and QMAP's DBAASP data."""
import hashlib
import json
import re
from sklearn.ensemble import ExtraTreesClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import DATA, ROOT, AA
from .features import features

BUNDLE = ROOT/'Models/target_screening.joblib'
TARGETS = {
    'E. coli': 'Escherichia coli',
    'S. aureus': 'Staphylococcus aureus',
    'P. aeruginosa': 'Pseudomonas aeruginosa',
    'K. pneumoniae': 'Klebsiella pneumoniae',
    'C. albicans': 'Candida albicans',
    'B. subtilis': 'Bacillus subtilis',
    'S. epidermidis': 'Staphylococcus epidermidis',
    'E. faecalis': 'Enterococcus faecalis',
    'M. tuberculosis': 'Mycobacterium tuberculosis (TB)',
    'SARS-CoV-2': 'SARS-CoV-2 (COVID-19)',
}
# Project comparison endpoint in GRAMPA log10 micromolar units.
CUTOFF = 1.0

QMAP_REVISION = 'cf5ff24fd963f293ff4bc24e5933b411bec1bf33'
QMAP_URL = ('https://huggingface.co/datasets/anthol42/qmap_benchmark_2025/resolve/'
            + QMAP_REVISION + '/dbaasp.json')


def qmap_labels(path):
    """Accept a class only when every supplied MIC summary supports it."""
    target_codes = {name.split(' (')[0]: code for code, name in TARGETS.items()}
    result = {code: {} for code in TARGETS}
    for record in json.loads(path.read_bytes()):
        sequence = record['sequence']
        if (record['nterminal'] or record['cterminal'] or record['bonds']
                or not re.fullmatch(f'[{AA}]{{5,200}}', sequence)):
            continue
        for species, measurements in record['targets'].items():
            if species not in target_codes:
                continue
            values = np.asarray(measurements, dtype=float)
            if not len(values) or not np.isfinite(values).all() or min(values) <= 0:
                continue
            classes = result[target_codes[species]].setdefault(sequence, set())
            classes.update((values <= 10).astype(int).tolist())
    return result


def estimator(name):
    if name == 'Logistic regression':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,random_state=42))
    return ExtraTreesClassifier(n_estimators=128, min_samples_leaf=2, max_features=.8,
                                n_jobs=1, random_state=42)


def metrics_for(model, test):
    scores = model.predict_proba(features(test.sequence))[:,1]
    return {'auc':float(roc_auc_score(test.label,scores)),
            'accuracy':float(accuracy_score(test.label,scores>=.5)),
            'balanced_accuracy':float(balanced_accuracy_score(test.label,scores>=.5))}, scores


def build():
    path = DATA/'raw/grampa.csv'
    qpath = DATA/'external/qmap_dbaasp.json'
    external = qmap_labels(qpath)
    raw = pd.read_csv(path)
    result = {}
    comparisons = []
    validation_rows = []
    for code, name in TARGETS.items():
        group = raw[raw.bacterium.eq(code)]
        records = group[group.unit.eq('uM') & group.datasource_has_modifications.eq(True)
                        & group.is_modified.eq(False) & np.isfinite(group.value)
                        & group.sequence.str.fullmatch(f'[{AA}]{{5,200}}', na=False)]
        observed = records.groupby('sequence').value.agg(['min','max','count']).reset_index()
        data = observed[(observed['max'] <= CUTOFF) | (observed['min'] > CUTOFF)].copy()
        data['label'] = (data['max'] <= CUTOFF).astype(int)
        seen = set(group.sequence)
        extra = pd.DataFrame([(seq,next(iter(labels))) for seq,labels in sorted(external[code].items())
                              if seq not in seen and len(labels)==1], columns=['sequence','label'])
        base = data[['sequence','label']]
        combined = pd.concat([base,extra],ignore_index=True).astype({'label':'int64'})
        counts = combined.label.value_counts().reindex([0,1],fill_value=0)
        entry = dict(name=name, records=len(group), sequences=len(combined), extra=len(extra),
                     grampa_sequences=len(base), qmap_reviewed=len(external[code]),
                     lower_mic=int(counts[1]), higher_mic=int(counts[0]))
        result[code] = entry
        if len(combined)<100 or counts.min()<25:
            print(code,'insufficient',len(combined), counts.tolist()); continue
        old_eligible = len(base)>=100 and base.label.value_counts().reindex([0,1],fill_value=0).min()>=25
        source = base if old_eligible else combined
        train,test = train_test_split(source,test_size=.25,stratify=source.label,random_state=42)
        fit,val = train_test_split(train,test_size=.2,stratify=train.label,random_state=42)
        baseline_pool = 'Original' if old_eligible else 'Expanded'
        pools = [(baseline_pool,train,fit)]
        if old_eligible and len(extra):
            pools.append(('Expanded',pd.concat([train,extra]),pd.concat([fit,extra])))
        trials = []
        for pool,all_train,subset in pools:
            for method in ['Logistic regression','Extra trees']:
                model = estimator(method).fit(features(subset.sequence),subset.label.astype(int))
                measured,_ = metrics_for(model,val)
                trials.append((measured['auc'],method,pool,all_train))
                validation_rows.append(dict(target=code, model=method, pool=pool,
                                            validation_rows=len(val), **measured))
        _,method,pool,all_train = max(trials,key=lambda trial:trial[0])
        assert not set(all_train.sequence) & set(test.sequence)
        model = estimator(method).fit(features(all_train.sequence),all_train.label.astype(int))
        measured,scores = metrics_for(model,test)
        baseline = estimator('Logistic regression').fit(features(train.sequence),train.label)
        before,_ = metrics_for(baseline,test)
        test = test.copy()
        test['target_activity_score'] = scores
        entry.update(model=model, test=test.reset_index(drop=True),train_count=len(all_train),
                     method=method,training_pool=pool,baseline_pool=baseline_pool,baseline=before,**measured)
        comparisons.append(dict(target=code,eligible_new=len(extra),model=method,pool=pool,
                                test_rows=len(test),**{'before_'+k:v for k,v in before.items()},**measured))
        print(code, len(extra), method, pool, before, measured, flush=True)
    audit = ROOT/'Results/final/target_benchmark'
    audit.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(comparisons).to_csv(audit/'comparison.csv',index=False)
    pd.DataFrame(validation_rows).to_csv(audit/'validation_trials.csv',index=False)
    provenance = dict(qmap_url=QMAP_URL,qmap_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
                      grampa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      method='Fixed GRAMPA test split for existing targets. Extra training sequences exclude every same-target GRAMPA sequence. Select training pool and model by inner validation AUC. Exact sequence split; related families can cross splits.')
    (audit/'sources.json').write_text(json.dumps(provenance,indent=2))
    temp = BUNDLE.with_suffix('.tmp')
    joblib.dump({'targets':result,**provenance,'raw_sha256':provenance['grampa_sha256']},temp,compress=3)
    temp.replace(BUNDLE)

def rank_target(frame, code, entry, ranking_rule, weight):
    scores = entry['model'].predict_proba(features(frame.sequence))[:,1]
    rows = frame[['sequence','activity_score','activity_prediction','toxicity_score','toxicity_prediction',
                  'rank','priority_score','prediction_engine']].rename(
        columns={'activity_score':'general_activity_score', 'activity_prediction':'general_activity_prediction',
                 'prediction_engine':'toxicity_engine', 'rank':'general_rank', 'priority_score':'general_priority_score'}).copy()
    rows.insert(0,'target',code)
    rows['target_activity_score'] = scores
    rows['priority_score'] = (100*scores*(1-rows.toxicity_score) if ranking_rule=='product'
                              else 100*(weight*scores+(1-weight)*(1-rows.toxicity_score)))
    rows['ranking_rule'] = ranking_rule
    if ranking_rule != 'product':
        rows['activity_weight'] = weight
    rows = rows.sort_values(['priority_score','sequence'],ascending=[False,True]).reset_index(drop=True)
    rows.insert(0,'rank',np.arange(1,len(rows)+1))
    return rows

if __name__=='__main__':
    build()
