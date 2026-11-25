"""Run: python -m streamlit run app.py"""
import json
from copy import copy
import pandas as pd
import streamlit as st
import altair as alt
from peptide_lab.config import MODELS, RESULTS, ROOT
from peptide_lab.features import parse_sequences
from peptide_lab.dashboard import dataset_page, history_page, save_run, inspection

from peptide_lab.tour import STOPS, show_tour

st.set_page_config(page_title='PeptideLab | Generate · Screen · Understand', page_icon='🧬', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fa; color:#132b40;}
h1,h2,h3 {color:#102d42; letter-spacing:-.025em;}
[data-testid="stSidebar"] {background:#102d42;}
[data-testid="stSidebar"] * {color:#edf5fa;}
[data-testid="stMetric"] {background:white; border:1px solid #dce6ed; border-radius:12px; padding:16px;}
.hero {background:linear-gradient(120deg,#102d42,#176b77);padding:28px 32px;border-radius:18px;color:white;margin-bottom:24px;}
.hero h1 {color:white;margin:0;font-size:2.3rem;}.hero p {color:#d1e8ea;margin-bottom:0;}
.stButton > button[kind="primary"] {background:#147d83;border:0;}
</style>''', unsafe_allow_html=True)
st.sidebar.title('🧬 PeptideLab')
st.sidebar.caption('STATISTICAL LEARNING · GROUP 4')
page = st.sidebar.radio('Workspace', list(STOPS), key='workspace')
st.sidebar.divider()
esm_ready = all((ROOT / f'Models/esm2_heads/{task}.pt').exists() for task in ['activity','toxicity'])
if esm_ready:
    st.sidebar.selectbox('Prediction engine', ['Compact baseline (fast)', 'ESM-2 + CNN'], key='prediction_engine_choice', help='Applies to new dataset, custom and generation predictions. Saved history retains its original engine.')
engine = 'esm2' if st.session_state.get('prediction_engine_choice') == 'ESM-2 + CNN' else 'compact'

st.sidebar.selectbox('Ranking rule', ['Product', 'Weighted difference'], key='rank_rule')
if st.session_state.rank_rule == 'Weighted difference':
    st.sidebar.slider('Activity weight', 0.0, 1.0, .5, .05, key='rank_weight')
st.sidebar.info('Research predictions only. Scores do not establish experimental activity or safety.')

def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))

@st.cache_resource
def load_pipeline(engine):
    from peptide_lab.pipeline import ScreeningPipeline
    return ScreeningPipeline(engine=engine)

def pipeline():
    # Share read-only model weights while keeping ranking settings local to this run.
    run = copy(load_pipeline(engine))
    run.ranking_rule = 'weighted_difference' if st.session_state.get('rank_rule') == 'Weighted difference' else 'product'
    run.activity_weight = st.session_state.get('rank_weight', .5)
    return run

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def table(frame):
    preferred = ['rank', 'sequence', 'priority_score', 'activity_score', 'activity_prediction', 'toxicity_score', 'toxicity_prediction', 'passes_property_screen', 'length', 'seen_in_grampa', 'seen_in_classifier_training']
    st.dataframe(frame[[c for c in preferred if c in frame]], hide_index=True, use_container_width=True, column_config={
        'priority_score': st.column_config.ProgressColumn('Priority / 100', min_value=0, max_value=100, format='%.1f'),
        'activity_score': st.column_config.NumberColumn('Activity score ↑', format='%.3f'),
        'toxicity_score': st.column_config.NumberColumn('Toxicity score ↓', format='%.3f'),
    })
    if 'prediction_engine' in frame and not frame.empty:
        saved = frame.iloc[0]
        rule = 'Product' if saved.ranking_rule == 'product' else f'Weighted difference (activity weight {saved.activity_weight:.2f})'
        st.caption(f"Last completed run · {saved.prediction_engine} · {rule}. Run again to apply new settings.")
    st.download_button('Download complete results · CSV', frame.to_csv(index=False).encode(), 'peptidelab_results.csv', 'text/csv')
    with st.expander('How to interpret these results'):
        st.write('Activity score: support for the antimicrobial class. Toxicity score: support for the toxic class. Both are model scores, not calibrated probabilities. Classification threshold = 0.5.')
        st.write('Product: 100*A*(1-T). Weighted difference: w*A-(1-w)*T, shifted by (1-w) and multiplied by 100 for display; sorting is unchanged. The CSV records the rule and weight.')
        st.write('Property screen: activity above its validation-selected cutoff AND toxicity below its validation-selected cutoff. All sequence-filter survivors remain ranked and exported, including property-screen failures. This flag is not experimental validation.')
        st.write('Priority is a transparent ranking rule, not a probability of clinical success. Novelty checks exact matches only. A sequence can resemble a known peptide without being an exact match.')
        st.write('The complete CSV includes model disagreement and simple sequence descriptors. Charge is K + R − D − E, an approximate descriptor rather than a pH-dependent charge calculation.')

ready = all((MODELS / name).exists() for name in ['activity_cnn.pt', 'toxicity_cnn.pt', 'vae.pt', 'activity_baseline.joblib', 'toxicity_baseline.joblib'])
if not ready:
    hero('Models are being prepared', 'The interface is ready. Finish training to enable prediction and generation.')
    st.code('python -m peptide_lab.training\npython -m peptide_lab.pipeline')
    st.stop()

show_tour(page)

if page == 'Overview':
    hero('PeptideLab', 'Explore general antimicrobial activity and toxicity, then compare the prediction for a particular organism.')
    a, t, v = read_json('activity_metrics.json'), read_json('toxicity_metrics.json'), read_json('vae_metrics.json')
    cols = st.columns(4)
    cols[0].metric('GRAMPA sequences', f"{read_json('preprocessing_audit.json')['canonical_unique_sequences']:,}")
    cols[1].metric('Activity · test ROC-AUC', f'{a["test"][a["selected_model"]]["roc_auc"]:.3f}')
    cols[2].metric('Toxicity · test ROC-AUC', f'{t["test"][t["selected_model"]]["roc_auc"]:.3f}')
    cols[3].metric('VAE training sequences', f'{v["splits"]["train"]:,}')
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader('Organism-specific screening')
        st.write('Compare general activity, toxicity and the selected organism’s MIC group for each peptide. Choose **Overall results** for test scores and data coverage.')
        st.button('Open target-specific screening', on_click=st.session_state.update, args=({'workspace':'Target-specific screening'},))
    with right:
        st.subheader('General screening')
        st.write('Use your own sequences, select measured peptides in **Dataset explorer**, or sample candidates in **Generate & rank**.')
        st.button('Screen your sequences', on_click=st.session_state.update, args=({'workspace':'Screen sequences'},))

elif page == 'Target-specific screening':
    st.title('Target-specific screening')
    from peptide_lab.target_dashboard import target_page
    target_page(pipeline, engine)

elif page == 'Dataset explorer':
    hero('Predict directly from the data.', 'Browse known peptides, run both predictors, and compare scores with available benchmark labels.')
    dataset_page(pipeline, table)

elif page == 'Prediction history':
    hero('Your recent prediction runs.', 'Revisit generated, custom, and dataset predictions and export complete results.')
    history_page(table)

elif page == 'Generate & rank':
    hero('Explore new candidates.', 'Sample the VAE, apply sequence-quality filters, and screen the accepted candidates with both predictors.')
    with st.form('generate'):
        c1, c2, c3 = st.columns(3)
        count = c1.slider('Raw sequences to sample', 100, 3000, 500, 100)
        st.caption('Every passed-filter peptide is scored, ranked, plotted and exported.')
        temperature = c2.slider('Sampling temperature', .5, 1.3, .9, .05, help='Lower values favor common residues; higher values increase sampling diversity.')
        seed = c3.number_input('Random seed', 0, 999999, 42)
        lengths = st.slider('Accepted peptide length', 5, 60, (10, 40))
        run = st.form_submit_button('Generate, screen & rank', type='primary')
    if run:
        with st.spinner('Sampling the VAE and evaluating both predictors…'):
            try:
                st.session_state.generated = pipeline().generate(count, int(seed), temperature, lengths[0], lengths[1])
                generated_frame, generated_audit = st.session_state.generated
                if not generated_frame.empty:
                    save_run("Generated candidates", generated_frame, generated_audit)
            except (ValueError, OSError, RuntimeError) as exc:
                st.session_state.pop('generated', None)
                st.error(str(exc))
    if 'generated' in st.session_state:
        frame, audit = st.session_state.generated
        cs = st.columns(3)
        cs[0].metric('Sampled', audit['sampled'])
        cs[1].metric('Passed filters', audit['passed_filters'])
        cs[2].metric('Ranked & exported', len(frame))
        if 'passed_property_screen' in audit:
            st.caption(f"Passed validation-selected property screen: {audit['passed_property_screen']} of {len(frame)}. All sequence-filter survivors remain available below.")
        if frame.empty:
            st.warning('No sequences passed the filters. Try a wider length range or another seed.')
        else:
            table(frame)
            st.subheader('Activity–toxicity trade-off')
            chart = alt.Chart(frame).mark_circle(size=70, opacity=.7).encode(
                x=alt.X('toxicity_score:Q', title='Toxicity score (lower preferred)', scale=alt.Scale(domain=[0,1])),
                y=alt.Y('activity_score:Q', title='Activity score (higher preferred)', scale=alt.Scale(domain=[0,1])),
                color=alt.Color('priority_score:Q', title='Priority / 100'),
                tooltip=[alt.Tooltip('rank:Q', title='Rank', format='d'), 'sequence:N',
                         alt.Tooltip('priority_score:Q', format='.3f'), alt.Tooltip('activity_score:Q', format='.4f'),
                         alt.Tooltip('toxicity_score:Q', format='.4f'), 'passes_property_screen:N', 'length:Q']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.caption(f'All {len(frame)} passed-filter peptides are plotted. Hover to see rank and sequence. Points may overlap.')
        with st.expander('Sampling audit and filters'):
            st.json(audit)
            st.write('Reject truncated outputs, out-of-range lengths, very repetitive sequences, exact known sequences, and duplicates. Report quality on the unfiltered pool separately under Model results.')

elif page == 'Screen sequences':
    hero('Inspect your sequences.', 'Paste amino-acid sequences or FASTA text. Download scores, predictions, and sequence descriptors.')
    sample = 'GILSGILGAGKSLVCGLSGLC\nKWKLFKKIGAVLKVL'
    with st.form('screen'):
        text = st.text_area('One sequence per line, or multiline FASTA', sample, height=160)
        upload = st.file_uploader('Or upload a FASTA / TXT / CSV file (CSV column: sequence)', type=['fasta', 'fa', 'txt', 'csv'])
        submit = st.form_submit_button('Screen sequences', type='primary')
    st.caption('Supported: 20 standard amino acids, 5–200 residues, up to 500 sequences. Example sequences are demonstration inputs, not new discoveries.')
    if submit:
        try:
            if upload is not None:
                if upload.size > 2_000_000:
                    raise ValueError('Upload a file smaller than 2 MB.')
                if upload.name.lower().endswith('.csv'):
                    df = pd.read_csv(upload)
                    if 'sequence' not in df:
                        raise ValueError('CSV must contain a sequence column.')
                    text = '\n'.join(df.sequence.astype(str))
                else:
                    text = upload.getvalue().decode('utf-8-sig')
            sequences = parse_sequences(text)
            with st.spinner('Running the activity and toxicity models…'):
                st.session_state.screened = pipeline().predict(sequences)
                save_run("Custom screening", st.session_state.screened, {"rows":len(sequences)})
        except (ValueError, UnicodeError, OSError) as exc:
            st.session_state.pop('screened', None)
            st.error(str(exc))
    if 'screened' in st.session_state:
        frame = st.session_state.screened
        table(frame)
        inspection(frame, 'custom_inspection')

elif page == 'Model results':
    hero('Model results', 'Compare held-out test scores, training curves and hyperparameter trials.')
    from peptide_lab.results_dashboard import results_page
    results_page()
