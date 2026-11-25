"""Compact, task-specific comparison of measured hyperparameter trials."""
import json
import pandas as pd
import streamlit as st
from .config import RESULTS
from .tuning import choose

NAMES = {'cnn': 'CNN', 'forest': 'Random forest', 'vae': 'VAE',
         'ensemble': 'Ensemble (CNN + Random forest)'}
PARAMETERS = {'lr': 'Learning rate', 'filters': 'Filters per kernel',
              'dropout': 'Dropout', 'n_estimators': 'Trees', 'max_depth': 'Maximum depth',
              'min_samples_leaf': 'Minimum leaf size', 'latent': 'Latent dimensions',
              'word_dropout': 'Word dropout'}

def value_text(value):
    return 'Unlimited' if value is None else str(value)

def changed_settings(config, baseline):
    changes = [f'{PARAMETERS[k]}: {value_text(baseline[k])} → {value_text(v)}'
               for k, v in config.items() if v != baseline[k]]
    return '; '.join(changes) if changes else 'Default settings (no changes)'

def comparison_table(rows, family):
    rows = sorted(rows, key=lambda r: r['trial'])
    baseline = next(r for r in rows if r['trial'] == 0)
    winner = choose(rows, family)
    metric = 'validation_objective' if family == 'vae' else 'validation_roc_auc'
    records = []
    for row in rows:
        record = {'Configuration': 'Default' if row['trial'] == 0 else f"Trial {row['trial']}",
                  'Changes from default': changed_settings(row['config'], baseline['config'])}
        record.update({PARAMETERS[k]: value_text(v) for k, v in row['config'].items()})
        if family == 'vae':
            record.update({'Validation reconstruction loss ↓': row['validation_reconstruction'],
                           'Validation token accuracy (%) ↑': 100*row['validation_token_accuracy'],
                           'Validation KL': row['validation_kl'],
                           'Validation objective ↓': row[metric]})
        else:
            record.update({'Validation accuracy (%) ↑': 100*row['validation_accuracy'],
                           'Validation precision ↑': row['validation_precision'],
                           'Validation recall ↑': row['validation_recall'],
                           'Validation F1 ↑': row['validation_f1'],
                           'Validation ROC-AUC ↑': row[metric]})
        record['Change in selection metric'] = row[metric] - baseline[metric]
        record['Verdict'] = 'Best tested (selected)' if row['trial'] == winner['trial'] else 'Not selected'
        records.append(record)
    return pd.DataFrame(records), baseline, winner

def test_comparison(folder, task, family):
    path = folder/'before_after.csv'
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_csv(path)
    if family == 'ensemble':
        results = json.loads((folder/f'{task}_test_results.json').read_text())
        source = pd.DataFrame([{'task': task, 'component': family, 'metric': metric,
                                'before': results['Before'][family][metric],
                                'after': results['After'][family][metric],
                                'delta': results['After'][family][metric]-results['Before'][family][metric]}
                               for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']])
    wanted = ['reconstruction', 'token_accuracy', 'objective', 'uniqueness', 'exact_novelty', 'natural_end_rate'] if family == 'vae' else ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']
    labels = {'reconstruction': 'Reconstruction loss ↓', 'token_accuracy': 'Token accuracy (%) ↑',
              'objective': 'Fixed objective ↓', 'uniqueness': 'Unique samples (%) ↑',
              'exact_novelty': 'Exact novelty vs GRAMPA (%) ↑', 'natural_end_rate': 'Natural termination (%) ↑',
              'accuracy': 'Accuracy (%) ↑', 'precision': 'Precision ↑', 'recall': 'Recall ↑',
              'f1': 'F1 ↑', 'roc_auc': 'ROC-AUC ↑', 'mcc': 'MCC ↑'}
    data = source[(source.task == task) & (source.component == family) & source.metric.isin(wanted)]
    output = []
    for metric in wanted:
        selected = data[data.metric == metric]
        if selected.empty:
            continue
        row = selected.iloc[0]
        scale = 100 if metric in ['accuracy', 'token_accuracy', 'uniqueness', 'exact_novelty', 'natural_end_rate'] else 1
        output.append({'Metric': labels[metric], 'Default': row['before']*scale,
                       'Selected configuration': row['after']*scale,
                       'Change': row['delta']*scale})
    return pd.DataFrame(output)

def ensemble_table(folder, task):
    """Use saved component validation predictions; never fit or select on test."""
    selection = json.loads((folder/'selection.json').read_text())[task]
    predictions = {}
    configurations = {}
    for stage in ['Default', 'Tuned components']:
        predictions[stage] = {}
        configurations[stage] = {}
        for family in ['cnn', 'forest']:
            trial = 0 if stage == 'Default' else selection[family]['trial']
            target = folder/task/family/f'trial_{trial:02d}'
            predictions[stage][family] = pd.read_csv(target/'validation_predictions.csv')
            configurations[stage][family] = json.loads((target/'config.json').read_text())
    from .training import metrics
    rows = []
    for stage, frames in predictions.items():
        cnn, forest = frames['cnn'], frames['forest']
        if not cnn[['sequence', 'label']].equals(forest[['sequence', 'label']]):
            raise ValueError('Ensemble validation predictions are not aligned by sequence and label.')
        values = metrics(cnn.label, (cnn.score.to_numpy()+forest.score.to_numpy())/2)
        configs = configurations[stage]
        changes = 'Default settings (no changes)' if stage == 'Default' else '; '.join(
            f'{NAMES[family]}: {changed_settings(configs[family], configurations["Default"][family])}'
            for family in ['cnn', 'forest'])
        rows.append({'Configuration': stage, 'Changes from default': changes,
                     'CNN settings': '; '.join(f'{PARAMETERS[k]}={value_text(v)}' for k,v in configs['cnn'].items()),
                     'Random forest settings': '; '.join(f'{PARAMETERS[k]}={value_text(v)}' for k,v in configs['forest'].items()),
                     'CNN weight': .5, 'Random forest weight': .5,
                     'Validation accuracy (%) ↑': values['accuracy']*100,
                     'Validation precision ↑': values['precision'], 'Validation recall ↑': values['recall'],
                     'Validation F1 ↑': values['f1'], 'Validation ROC-AUC ↑': values['roc_auc']})
    baseline = rows[0]['Validation ROC-AUC ↑']
    best = max(range(len(rows)), key=lambda i: rows[i]['Validation ROC-AUC ↑'])
    for i, row in enumerate(rows):
        row['Change in selection metric'] = row['Validation ROC-AUC ↑']-baseline
        row['Verdict'] = 'Best compared ensemble' if i == best else 'Not selected'
    return pd.DataFrame(rows), selection

def ensemble_view(folder, task):
    st.write('**Activity score = 0.5 × CNN score + 0.5 × random-forest score**')
    st.caption('Two ensemble configurations are compared: default components and validation-selected tuned components. The averaging weights stay fixed; no separate ensemble-weight or exhaustive component-pair search was run.')
    frame, selection = ensemble_table(folder, task)
    st.markdown('**Default ensemble vs ensemble with tuned components**')
    show_table(frame)
    st.download_button('Download comparison table', frame.to_csv(index=False), task+'_ensemble_tuning.csv',
                       'text/csv', key='tuning_download_trials_'+task)
    st.markdown('**Final verdict**')
    before, after = frame['Validation ROC-AUC ↑'].tolist()
    if after > before:
        st.success(f'The ensemble with tuned components has higher validation ROC-AUC: {before:.6f} → {after:.6f}. See the changes column for the component settings.')
    else:
        st.success(f'The default ensemble remains best among these two configurations (validation ROC-AUC {before:.6f}).')
    model = selection['pipeline']['after']['model']
    st.write(f'**Best overall activity model:** {NAMES[model]}, selected by comparing CNN, random forest and ensemble validation ROC-AUC.')
    st.markdown('**Test results: default ensemble vs ensemble with tuned components**')
    comparison = test_comparison(folder, task, 'ensemble')
    show_table(comparison)
    st.download_button('Download before and after results', comparison.to_csv(index=False), task+'_ensemble_before_after.csv',
                       'text/csv', key='tuning_download_comparison_'+task)
    st.caption('Test results describe both ensembles and did not choose the settings. A higher validation ROC-AUC does not guarantee higher test accuracy. Percentage changes are percentage points. This is a single-seed comparison on existing benchmarks. Normal predictions still use the original deployed ensemble.')

def show_table(frame):
    formats = {col: st.column_config.NumberColumn(col, format='%.6f')
               for col in frame.select_dtypes(include='number').columns}
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config=formats)

def tuning_page(task):
    folder = RESULTS/'tuning'
    status_path = folder/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'state': 'not started'}
    if status['state'] != 'complete':
        st.info('Tuning status: ' + status['state'])
        st.button('Refresh tuning results', key='tuning_refresh_'+task)
        if status['state'] == 'failed':
            st.error(status.get('error', 'Search failed.'))
    available = {family: [json.loads(p.read_text()) for p in sorted((folder/task/family).glob('trial_*/result.json'))]
                 for family in (['vae'] if task == 'vae' else ['cnn', 'forest'])}
    available = {family: rows for family, rows in available.items() if rows}
    if task == 'activity' and status['state'] == 'complete' and (folder/'selection.json').exists():
        available['ensemble'] = []
    if not available:
        st.info('Completed tuning trials will appear here when available.')
        return
    family = st.selectbox('Model to compare', list(available), format_func=lambda f: NAMES[f], key='tuning_family_'+task)
    if family == 'ensemble':
        ensemble_view(folder, task)
        return
    rows = available[family]
    frame, baseline, winner = comparison_table(rows, family)
    st.markdown('**Default settings compared with every tested configuration**')
    st.caption('All rows below are validation results on the same split. “Changes from default” lists exactly what changed in each trial. Multiple changes in one trial are a configuration comparison, not isolated proof of each parameter’s effect.')
    show_table(frame)
    st.download_button('Download comparison table', frame.to_csv(index=False), task+'_'+family+'_tuning.csv',
                       'text/csv', key='tuning_download_trials_'+task)
    key = 'validation_objective' if family == 'vae' else 'validation_roc_auc'
    metric_name = 'validation objective (lower is better)' if family == 'vae' else 'validation ROC-AUC (higher is better)'
    if status['state'] == 'complete':
        title = 'Final verdict'
    else:
        title = 'Provisional verdict — completed trials only'
    st.markdown('**'+title+'**')
    if winner['trial'] == 0:
        st.success(f"{NAMES[family]}: keep the default settings. None of the tested alternatives improved the selection metric. Default {metric_name}: {winner[key]:.6f}.")
    else:
        st.success(f"{NAMES[family]}: Trial {winner['trial']} was best among the tested configurations. {changed_settings(winner['config'], baseline['config'])}. The {metric_name} changed from {baseline[key]:.6f} to {winner[key]:.6f}.")
    selections_path = folder/'selection.json'
    if status['state'] == 'complete' and selections_path.exists():
        selections = json.loads(selections_path.read_text())
        if task in selections and 'pipeline' in selections[task]:
            selected = selections[task]['pipeline']['after']['model']
            labels = {'cnn': 'CNN', 'forest': 'random forest', 'ensemble': 'CNN + random-forest ensemble'}
            st.write(f"**Best overall model for {task}:** {labels[selected]}, chosen using validation ROC-AUC. The family verdict above applies to {NAMES[family]} specifically.")
    st.markdown('**Test results: default vs selected configuration**')
    if status['state'] == 'complete':
        comparison = test_comparison(folder, task, family)
        if not comparison.empty:
            show_table(comparison)
            st.download_button('Download before and after results', comparison.to_csv(index=False),
                               task+'_'+family+'_before_after.csv', 'text/csv', key='tuning_download_comparison_'+task)
        st.caption('Test data did not select the winner. A validation improvement may not improve every test metric. Changes in percentage rows are percentage points. VAE sampling compares separate 500-sample pools under identical sampling settings.')
    else:
        st.info('Final test comparison appears after the full search finishes.')
    st.caption('Best means best among four tested configurations per family, using one training seed. Existing benchmarks have known similarity/length bias. Normal prediction pages still use the original deployed models; these tables evaluate saved tuning checkpoints.')
