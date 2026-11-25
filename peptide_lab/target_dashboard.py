"""Target selection, measured benchmarks and batch screening."""
import json
import altair as alt
import joblib
import pandas as pd
import streamlit as st
from .config import RESULTS
from .dashboard import save_run
from .features import parse_sequences
from .targets import BUNDLE, rank_target

@st.cache_resource
def load_targets(timestamp):
    return joblib.load(BUNDLE)['targets']

def target_result_table(result):
    st.caption('Each rank is within this batch. General rank combines general activity and toxicity. Target rank uses the organism activity score and the same toxicity score.')
    columns = ['rank', 'general_rank', 'sequence', 'general_activity_score', 'target_activity_score', 'toxicity_score']
    if 'observed_lower_MIC_label' in result:
        columns.append('observed_lower_MIC_label')
    st.dataframe(result, column_order=columns + [col for col in result if col not in columns],
                 hide_index=True, use_container_width=True, column_config={
                     'rank':'Target rank', 'general_rank':'General rank', 'general_activity_score':'General activity score',
                     'target_activity_score':'Organism activity score', 'toxicity_score':'General toxicity score',
                     'observed_lower_MIC_label':st.column_config.NumberColumn('Measured lower MIC', help='1 means measured MIC at or below 10 µM. 0 means above 10 µM.')})
    code = result['target'].iloc[0]
    st.download_button('Download target results', result.to_csv(index=False), f'{code.replace(" ","_")}_screening.csv','text/csv')
    st.subheader(f'{code} activity and toxicity')
    tooltip = [
        alt.Tooltip('sequence:N', title='Peptide'),
        alt.Tooltip('rank:Q', title='Target rank', format='d'),
        alt.Tooltip('general_rank:Q', title='General rank', format='d'),
        alt.Tooltip('target_activity_score:Q', title=f'{code} activity', format='.3f'),
        alt.Tooltip('general_activity_score:Q', title='General activity', format='.3f'),
        alt.Tooltip('toxicity_score:Q', title='General toxicity', format='.3f'),
        alt.Tooltip('priority_score:Q', title='Target priority / 100', format='.1f'),
    ]
    if 'observed_lower_MIC_label' in result:
        tooltip.append(alt.Tooltip('observed_lower_MIC_label:Q', title='Measured lower MIC (1 = yes)', format='d'))
    chart = alt.Chart(result).mark_circle(size=75, opacity=.75).encode(
        x=alt.X('toxicity_score:Q', title='General toxicity score', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('target_activity_score:Q', title=f'{code} activity score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('priority_score:Q', title='Target priority / 100', scale=alt.Scale(domain=[0, 100])),
        tooltip=tooltip,
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption(f'All {len(result)} peptides in the table are plotted. Hover for ranks and scores. Peptides with identical scores share a position.')

def overall_results(targets):
    st.subheader('General predictors')
    rows = []
    for task in ('activity', 'toxicity'):
        report = json.loads((RESULTS / f'{task}_metrics.json').read_text(encoding='utf-8'))
        model = report['selected_model']
        scores = report['test'][model]
        rows.append({'Task': task.capitalize(), 'Compact model': model, 'Test peptides': sum(map(sum, scores['confusion_matrix'])),
                     'Accuracy': scores['accuracy'], 'ROC-AUC': scores['roc_auc']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        'Accuracy': st.column_config.NumberColumn(format='%.3f'), 'ROC-AUC': st.column_config.NumberColumn(format='%.3f')})
    st.button('Run a general prediction', on_click=st.session_state.update, args=({'workspace': 'Screen sequences'},))

    st.subheader('Organism-specific predictors')
    rows = []
    for entry in targets.values():
        tested = 'test' in entry
        rows.append({'Organism': entry['name'], 'Status': 'Evaluated' if tested else 'Evidence only',
                     'Eligible sequences': entry['sequences'], 'Test peptides': len(entry['test']) if tested else None,
                     'Model': entry.get('method'), 'Accuracy': entry.get('accuracy'), 'ROC-AUC': entry.get('auc')})
    summary = pd.DataFrame(rows)
    evaluated = summary[summary.Status == 'Evaluated']
    cols = st.columns(3)
    cols[0].metric('Evaluated organisms', f'{len(evaluated)} / {len(summary)}')
    if not evaluated.empty:
        cols[1].metric('Mean test accuracy', f"{evaluated['Accuracy'].mean():.1%}")
        cols[2].metric('Mean test ROC-AUC', f"{evaluated['ROC-AUC'].mean():.3f}")
    st.caption('Each evaluated organism counts once in the average. Each model has its own held-out test set. The general activity and toxicity benchmarks use different labels and test sets.')
    st.dataframe(summary, hide_index=True, column_config={
        'Accuracy': st.column_config.NumberColumn(format='%.3f'), 'ROC-AUC': st.column_config.NumberColumn(format='%.3f'),
        'Test peptides': st.column_config.NumberColumn(format='%d')})
    st.download_button('Download organism comparison', summary.to_csv(index=False), 'organism_comparison.csv', 'text/csv')

def target_page(pipeline):
    if not BUNDLE.exists():
        st.info('Prepare the target models with python -m peptide_lab.targets.')
        return
    targets = load_targets(BUNDLE.stat().st_mtime_ns)
    view = st.radio('View', ['Screen one organism', 'Overall results'], horizontal=True, key='target_view')
    if view == 'Overall results':
        overall_results(targets)
        return
    code = st.selectbox('Target organism', list(targets), key='target_organism',
                        format_func=lambda k: targets[k]['name'] + (' (insufficient data)' if 'model' not in targets[k] else ''))
    entry = targets[code]
    if code == 'SARS-CoV-2' and 'model' not in entry:
        st.info('No viral assay model is available. GRAMPA and the bacterial MIC measurements in QMAP provide no SARS-CoV-2 labels.')
        st.caption('The 2022 ACovPepDB study reported 50 peptides with activity against SARS-CoV-2. Its full coronavirus collection spans 18 assay types. These are published evidence counts. Viral inhibition and bacterial MIC measure different endpoints. [ACovPepDB publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9192597/)')
        return
    st.caption('The target score estimates whether the recorded minimum inhibitory concentration (MIC) is at or below 10 µM for this organism. The sidebar engine supplies general activity and toxicity scores.')
    if 'model' not in entry:
        st.info(f"{entry['grampa_sequences']} eligible GRAMPA sequences and {entry['extra']} additional QMAP sequences give {entry['sequences']} in total. "
                f"{entry['lower_mic']} are in the lower-MIC group and {entry['higher_mic']} are above the cutoff. "
                "The project requires 100 sequences overall and 25 in each group to train a model.")
        return
    with st.expander('Evidence for this target'):
        pool = 'GRAMPA + QMAP' if entry['training_pool'] == 'Expanded' else 'GRAMPA'
        baseline_pool = 'GRAMPA + QMAP' if entry['baseline_pool'] == 'Expanded' else 'GRAMPA'
        st.write(f"{entry['grampa_sequences']:,} GRAMPA sequences qualified. QMAP contributed {entry['extra']:,} additional eligible sequences. "
                 f"The selected model trained on {entry['train_count']:,} sequences from {pool}. Both models were tested on the same {len(entry['test']):,} held-out sequences.")
        results = []
        for label, source, scores in [('Baseline · Logistic regression', baseline_pool, entry['baseline']),
                                      ('Selected · ' + entry['method'], pool, entry)]:
            results.append({'Model': label, 'Training data': source, 'ROC-AUC': round(scores['auc'], 3),
                            'Accuracy': f"{scores['accuracy']:.1%}", 'Balanced accuracy': f"{scores['balanced_accuracy']:.1%}"})
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
        st.caption('Both models use 25 sequence descriptors. Validation ROC-AUC chose the model and training data. '
                   'Test accuracy counts correct classifications at a score cutoff of 0.5. ROC-AUC measures how well the scores order the two MIC groups. '
                   'Identical sequences are separated between training and test, but related peptides may occur in both. '
                   'A score for one organism does not establish selectivity against other organisms.')
    mode = st.radio('Peptide source',['Dataset peptides','Enter sequences','Generate peptides'],horizontal=True,key='target_mode')
    test = entry['test']
    if mode=='Dataset peptides':
        seqs = st.multiselect('Choose measured test peptides',test.sequence.tolist(),default=test.sequence.head(3).tolist(),key='target_dataset_'+code)
    elif mode=='Enter sequences':
        raw = st.text_area('Sequences or FASTA',key='target_input')
    else:
        st.caption('The existing VAE generates the initial pool. The selected organism determines the activity score and final ranking.')
        count = st.slider('Raw sample count',100,1000,100,100)
        seed = st.number_input('Sampling seed',0,999999,42)
    result_key = 'target_last_result'
    if st.button('Screen for selected target',type='primary'):
        st.session_state.pop(result_key,None)
        try:
            pipe = pipeline()
            if mode=='Generate peptides':
                frame, audit = pipe.generate(count=count,seed=int(seed))
                if frame.empty:
                    raise ValueError('No generated peptides passed the sequence filters.')
            else:
                sequences = parse_sequences(raw) if mode=='Enter sequences' else seqs
                frame = pipe.predict(sequences)
            result = rank_target(frame,code,entry,pipe.ranking_rule,pipe.activity_weight)
            if mode=='Dataset peptides':
                result = result.merge(test[['sequence','label']].rename(columns={'label':'observed_lower_MIC_label'}),on='sequence',validate='one_to_one')
            st.session_state[result_key] = (code,mode,result)
            settings = {'organism': code, 'source': mode, 'general_engine': pipe.engine, 'ranking_rule': pipe.ranking_rule}
            if pipe.ranking_rule != 'product':
                settings['activity_weight'] = pipe.activity_weight
            if mode == 'Generate peptides':
                settings.update(raw_sample_count=count, seed=int(seed), passed_filters=audit['passed_filters'])
            save_run('Target screening · ' + code, result, settings)
        except ValueError as exc:
            st.warning(str(exc))
    saved = st.session_state.get(result_key)
    if saved and saved[:2]==(code,mode):
        result = saved[2]
        st.subheader(f"Last completed screening for {entry['name']}")
        target_result_table(result)
