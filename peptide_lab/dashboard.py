"""Additional dashboard pages; training stays outside the user interface."""
from datetime import datetime
import pandas as pd
import streamlit as st
from .datasets import catalog, browse, attach_labels, SOURCES

@st.cache_data
def load_catalog():
    return catalog()

def save_run(kind, frame, settings):
    runs = st.session_state.setdefault('prediction_runs', [])
    runs.append({'kind': kind, 'time': datetime.now().strftime('%H:%M:%S'), 'settings': settings, 'frame': frame.copy()})
    del runs[:-10]

def inspection(frame, key):
    st.subheader('Inspect an individual prediction')
    sequence = st.selectbox('Peptide to inspect', frame.sequence.tolist(), key=key)
    row = frame.set_index('sequence').loc[sequence]
    st.code(sequence, language=None)
    cols = st.columns(3)
    cols[0].metric('Activity score', f'{row.activity_score:.3f}', help='Support for antimicrobial class; not MIC.')
    cols[1].metric('Toxicity score', f'{row.toxicity_score:.3f}', help='Support for toxic benchmark class; not a safety certificate.')
    cols[2].metric('Priority / 100', f'{row.priority_score:.1f}')
    if 'rank' in row:
        st.caption(f'Rank {int(row["rank"])} of {len(frame)} in this run.')
    st.write(f'**Predicted activity:** {row.activity_prediction} · **Predicted toxicity:** {row.toxicity_prediction}')
    for task in ['activity', 'toxicity']:
        label = row.get(task + '_label', pd.NA)
        split = row.get(task + '_split', 'unknown')
        if pd.notna(label):
            meaning = ('Antimicrobial' if int(label) else 'Non-antimicrobial') if task=='activity' else ('Toxic' if int(label) else 'Non-toxic')
            matches = bool(row[task + '_matches_label'])
            st.write(f'**Known {task} label:** {meaning} · **Dataset split:** {split} · **Matches prediction:** {"Yes" if matches else "No"}')
        elif task + '_label' in row:
            st.caption(f'No {task} benchmark label is available for this sequence. A prediction is not a measured label.')
    with st.expander('Sequence properties and predictor agreement'):
        st.json({name: float(row[name]) for name in ['length','charge_proxy','hydrophobic_fraction','mean_hydropathy','entropy','activity_model_disagreement','toxicity_model_disagreement']})
        st.caption('Disagreement is the absolute difference between CNN and forest scores, not a calibrated uncertainty interval.')

def dataset_page(pipeline, table):
    st.write('Choose existing peptides and run both predictors. Dataset labels are shown only where available; GRAMPA membership does not supply a toxicity label.')
    c1, c2, c3 = st.columns(3)
    source = c1.selectbox('Dataset', SOURCES, key='dataset_source')
    explanations = {'GRAMPA':'Generator source pool. Benchmark activity/toxicity labels may be unavailable.', 'Activity benchmark':'Known labels: 1 = antimicrobial, 0 = non-antimicrobial. Both properties are predicted.', 'Toxicity benchmark':'Known labels: 1 = toxic, 0 = non-toxic. Both properties are predicted.'}
    st.info(explanations[source])
    split = c2.selectbox('Source dataset split', ['All','test','validation','train'], disabled=source=='GRAMPA', key='dataset_split')
    label = c3.selectbox('Known source label', ['All','1','0'], disabled=source=='GRAMPA', key='dataset_label', help='Activity: 1=antimicrobial. Toxicity: 1=toxic.')
    c1, c2 = st.columns([2,1])
    query = c1.text_input('Find a sequence or residue fragment', key='dataset_search')
    lengths = c2.slider('Dataset peptide length',5,200,(5,200),key='dataset_lengths')
    filtered = browse(load_catalog(), source, split, label, *lengths, query)
    st.metric('Matching dataset peptides', len(filtered))
    st.caption('Rows from training or validation sets are useful for demonstration, not independent performance estimates. Each result includes separate split membership for each predictor.')
    if filtered.empty:
        st.info('No sequences match. Clear the search or widen the filters.')
        return
    page_number = st.number_input('Browse page (50 rows per page)', 1, max(1,(len(filtered)+49)//50),1,key=f'dataset_page_{source}_{split}_{label}_{len(filtered)}')
    view = filtered.iloc[(page_number-1)*50:page_number*50]
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button('Download filtered dataset', filtered.to_csv(index=False).encode(), 'filtered_dataset.csv','text/csv')
    mode = st.selectbox('Prediction batch', ['Selected peptides','Random sample of filtered dataset','First rows of filtered dataset'], key='dataset_batch_mode')
    with st.form('dataset_predict'):
        selection, count, seed = [], 25, 42
        if mode == 'Selected peptides':
            st.caption('Choose the individual sequences to predict from the current 50-row page.')
            selection = st.multiselect('Select peptides from this page', view.sequence.tolist(), default=view.sequence.head(3).tolist())
        else:
            count = st.number_input('Number of peptides to predict', 1, 500, 25)
            if mode.startswith('Random'):
                seed = st.number_input('Random sample seed', 0, 999999, 42)
                st.caption('Random sampling uses all matching rows across pages. The same filters and seed reproduce the selection.')
            else:
                st.caption('Uses the first matching rows across pages; no manual selection or random seed is needed.')
        run = st.form_submit_button('Predict dataset activity & toxicity',type='primary')
    if run:
        if mode=='Selected peptides':
            chosen = filtered[filtered.sequence.isin(selection)]
        elif mode.startswith('Random'):
            chosen = filtered.sample(n=min(int(count),len(filtered)),random_state=int(seed))
        else:
            chosen = filtered.head(int(count))
        if chosen.empty:
            st.warning('Select at least one peptide.')
            st.session_state.pop('dataset_result',None)
        else:
            with st.spinner('Running activity and toxicity inference on dataset sequences…'):
                result = attach_labels(pipeline().predict(chosen.sequence.tolist()),chosen)
            settings = {'dataset':source,'source_split':split,'known_label_filter':label,'mode':mode,'seed':int(seed),'rows':len(result)}
            st.session_state.dataset_result = (result,settings)
            save_run('Dataset screening',result,settings)
    if 'dataset_result' in st.session_state:
        result,settings = st.session_state.dataset_result
        st.divider()
        st.subheader('Dataset prediction results')
        st.caption(f'Last completed run: {settings["dataset"]} · source split {settings["source_split"]} · {len(result)} peptides. Changing filters does not change these results until you run prediction again.')
        table(result)
        columns = ['sequence','activity_label','activity_prediction','activity_matches_label','activity_split','toxicity_label','toxicity_prediction','toxicity_matches_label','toxicity_split']
        st.markdown('**Known labels versus model predictions**')
        comparison = result[columns].copy()
        # Streamlit renders nullable Boolean cells as unchecked boxes. Explicit
        # text avoids showing a missing ground truth as a failed prediction.
        for task in ['activity','toxicity']:
            comparison[task+'_matches_label'] = comparison[task+'_matches_label'].map(lambda value: 'Not available' if pd.isna(value) else ('Yes' if value else 'No'))
        st.dataframe(comparison,hide_index=True,use_container_width=True)
        st.caption('Blank means no benchmark label exists. A correct prediction on a training sequence is not test evidence.')
        inspection(result,'dataset_inspection')

def history_page(table):
    runs = st.session_state.get('prediction_runs',[])
    if not runs:
        st.info('Run generation, dataset screening, or custom screening first. Your latest 10 runs are kept in this browser session.')
        return
    choice = st.selectbox('Prediction run',range(len(runs)),index=len(runs)-1,format_func=lambda i:f'{i+1}. {runs[i]["kind"]} · {runs[i]["time"]} · {len(runs[i]["frame"])} peptides')
    run = runs[choice]
    st.json(run['settings'])
    if 'target_activity_score' in run['frame']:
        from .target_dashboard import target_result_table
        target_result_table(run['frame'])
    else:
        table(run['frame'])
        inspection(run['frame'],'history_inspection')
    st.caption('History is session-local and clears on a new session. Download CSVs to retain your work.')
