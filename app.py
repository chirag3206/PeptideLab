"""Run: python -m streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from peptide_lab.config import MODELS, RESULTS, ROOT
from peptide_lab.features import parse_sequences
from peptide_lab.dashboard import dataset_page, history_page, research_page, save_run, inspection

st.set_page_config(page_title='PeptideLab | Generate · Screen · Understand', page_icon='🧬', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fa; color:#132b40;}
h1,h2,h3 {color:#102d42; letter-spacing:-.025em;}
[data-testid="stSidebar"] {background:#102d42;}
[data-testid="stSidebar"] * {color:#edf5fa;}
[data-testid="stMetric"] {background:white; border:1px solid #dce6ed; border-radius:12px; padding:16px;}
.hero {background:linear-gradient(120deg,#102d42,#176b77);padding:28px 32px;border-radius:18px;color:white;margin-bottom:24px;}
.hero h1 {color:white;margin:0;font-size:2.3rem;}.hero p {color:#d1e8ea;margin-bottom:0;}
.eyebrow {font-size:12px;letter-spacing:2px;font-weight:700;color:#89d6c7;}
.stButton > button[kind="primary"] {background:#147d83;border:0;}
</style>''', unsafe_allow_html=True)
st.sidebar.title('🧬 PeptideLab')
st.sidebar.caption('STATISTICAL LEARNING · GROUP 4')
page = st.sidebar.radio('Workspace', ['Overview', 'Dataset explorer', 'Generate & rank', 'Screen sequences', 'Prediction history', 'Model results', 'Research checks', 'Learn the project'])
st.sidebar.divider()
st.sidebar.caption('Local inference · CPU ready\n\nVAE generation + activity and toxicity screening')
st.sidebar.info('Research predictions only. Scores do not establish experimental activity or safety.')

def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))

@st.cache_resource
def pipeline():
    from peptide_lab.pipeline import ScreeningPipeline
    return ScreeningPipeline()

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">PEPTIDELAB / RESEARCH WORKSPACE</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def table(frame):
    preferred = ['sequence', 'priority_score', 'activity_score', 'activity_prediction', 'toxicity_score', 'toxicity_prediction', 'length', 'seen_in_grampa', 'seen_in_classifier_training']
    st.dataframe(frame[[c for c in preferred if c in frame]], hide_index=True, use_container_width=True, column_config={
        'priority_score': st.column_config.ProgressColumn('Priority / 100', min_value=0, max_value=100, format='%.1f'),
        'activity_score': st.column_config.NumberColumn('Activity score ↑', format='%.3f'),
        'toxicity_score': st.column_config.NumberColumn('Toxicity score ↓', format='%.3f'),
    })
    st.download_button('Download complete results · CSV', frame.to_csv(index=False).encode(), 'peptidelab_results.csv', 'text/csv')
    with st.expander('How to interpret these results'):
        st.write('Activity score: support for the antimicrobial class. Toxicity score: support for the toxic class. Both are model scores, not calibrated probabilities. Classification threshold = 0.5.')
        st.code('priority = 100 × activity_score × (1 − toxicity_score)', language=None)
        st.write('Priority is a transparent ranking rule, not a probability of clinical success. Novelty checks exact matches only. A sequence can resemble a known peptide without being an exact match.')
        st.write('The complete CSV includes model disagreement and simple sequence descriptors. Charge is K + R − D − E, an approximate descriptor rather than a pH-dependent charge calculation.')

ready = all((MODELS / name).exists() for name in ['activity_cnn.pt', 'toxicity_cnn.pt', 'vae.pt', 'activity_baseline.joblib', 'toxicity_baseline.joblib'])
if not ready:
    hero('Models are being prepared', 'The interface is ready. Finish training to enable prediction and generation.')
    st.code('python -m peptide_lab.training\npython -m peptide_lab.pipeline')
    st.stop()

if page == 'Overview':
    hero('From sequences to a shortlist.', 'Generate peptide candidates, screen two biological properties, and explore the evidence behind every result.')
    a, t, v = read_json('activity_metrics.json'), read_json('toxicity_metrics.json'), read_json('vae_metrics.json')
    cols = st.columns(4)
    cols[0].metric('GRAMPA sequences', '6,760')
    cols[1].metric('Activity · test ROC-AUC', f'{a["test"][a["selected_model"]]["roc_auc"]:.3f}')
    cols[2].metric('Toxicity · test ROC-AUC', f'{t["test"][t["selected_model"]]["roc_auc"]:.3f}')
    cols[3].metric('VAE training sequences', f'{v["splits"]["train"]:,}')
    st.subheader('One connected workflow')
    cols = st.columns(4)
    for col, title, desc in zip(cols, ['01 · Generate', '02 · Predict activity', '03 · Predict toxicity', '04 · Rank & export'], ['An LSTM VAE learns patterns in GRAMPA and samples new sequences.', 'A trained classifier scores membership in the antimicrobial class.', 'A separate classifier estimates membership in the toxic class.', 'A clear scoring rule creates a downloadable shortlist.']):
        with col:
            st.markdown('**' + title + '**')
            st.write(desc)
    st.divider()
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader('Ready for the demonstration')
        st.write('Use **Dataset explorer** to predict existing peptides and compare known labels, **Generate & rank** to create candidates, or **Screen sequences** for your own input. Revisit runs under **Prediction history**. Open **Model results** for held-out metrics and **Learn the project** for the viva explanation.')
        if (RESULTS / 'ranked_candidates.csv').exists():
            st.caption('Saved reproducible demonstration run · seed 42')
            table(pd.read_csv(RESULTS / 'ranked_candidates.csv').head(5))
    with right:
        st.subheader('What this project claims')
        st.write('A working computational prototype with reproducible training, held-out evaluation, and candidate prioritization.')
        st.write('Activity is binary classification, not species-specific MIC prediction. Toxicity is a benchmark label, not a comprehensive safety assessment. Similar sequences may occur across dataset splits.')

elif page == 'Dataset explorer':
    hero('Predict directly from the data.', 'Browse known peptides, run both predictors, and compare scores with available benchmark labels.')
    dataset_page(pipeline, table)

elif page == 'Prediction history':
    hero('Your recent prediction runs.', 'Revisit generated, custom, and dataset predictions and export complete results.')
    history_page(table)

elif page == 'Research checks':
    hero('Inspect the research evidence.', 'Reproducible preprocessing, latent diagnostics, baseline checks, and ranking sensitivity.')
    research_page()

elif page == 'Generate & rank':
    hero('Explore new candidates.', 'Sample the VAE, apply sequence-quality filters, and screen the accepted candidates with both predictors.')
    with st.form('generate'):
        c1, c2, c3 = st.columns(3)
        count = c1.slider('Candidates requested', 5, 100, 30, 5)
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
                st.error(str(exc))
    if 'generated' in st.session_state:
        frame, audit = st.session_state.generated
        cs = st.columns(3)
        cs[0].metric('Sampled', audit['sampled'])
        cs[1].metric('Passed filters', audit['accepted_before_limit'])
        cs[2].metric('Returned', len(frame))
        if len(frame) < audit['requested']:
            st.warning('The fixed sampling pool produced fewer eligible sequences than requested. Try a wider length range or another seed.')
        if not frame.empty:
            table(frame)
            st.subheader('Activity–toxicity trade-off')
            st.scatter_chart(frame, x='toxicity_score', y='activity_score', color='priority_score', x_label='Toxicity score · lower preferred', y_label='Activity score · higher preferred')
        with st.expander('Sampling audit and filters'):
            st.json(audit)
            st.write('Reject truncated outputs, out-of-range lengths, very repetitive sequences, exact known sequences, and duplicates. Report quality on the unfiltered pool separately under Model results.')
    else:
        st.info('Choose settings and run the complete generation → activity → toxicity → ranking pipeline.')

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
        with st.expander('Sequence descriptors and model disagreement'):
            st.dataframe(frame, hide_index=True, use_container_width=True)

elif page == 'Model results':
    hero('Evidence, not just outputs.', 'Held-out test results, baseline comparisons, training curves, and explicit limitations.')
    tabs = st.tabs(['Activity predictor', 'Toxicity predictor', 'VAE generator', 'Data provenance'])
    for tab, task in zip(tabs[:2], ['activity', 'toxicity']):
        with tab:
            report = read_json(task + '_metrics.json')
            st.write(f'**Selected model:** {report["selected_model"]} · chosen by validation ROC-AUC · threshold 0.5')
            rows = [{"model": name, **{k: v for k, v in values.items() if k != 'confusion_matrix'}} for name, values in report['test'].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('**Training / validation**')
                history = pd.read_csv(RESULTS / f'{task}_history.csv').set_index('epoch')
                st.line_chart(history[['validation_auc']])
                st.caption('Validation ROC-AUC across epochs; best checkpoint retained.')
            with c2:
                st.markdown('**Test confusion matrix**')
                matrix = report['test'][report['selected_model']]['confusion_matrix']
                st.dataframe(pd.DataFrame(matrix, index=['Actual negative', 'Actual positive'], columns=['Predicted negative', 'Predicted positive']))
                st.caption('Positive means antimicrobial for activity; toxic for toxicity.')
            st.markdown('**Interpretable baseline · feature importance**')
            importance = pd.read_csv(RESULTS / f'{task}_feature_importance.csv').head(10).set_index('feature')
            st.bar_chart(importance)
            st.caption('Random-forest impurity importance; descriptive association, not a causal explanation.')
            st.info(report['limitations'])
            with st.expander('Dataset audit / split counts'):
                st.json(read_json(task + '_data_audit.json'))
    with tabs[2]:
        report = read_json('vae_metrics.json')
        cols = st.columns(3)
        cols[0].metric('Unique samples / 500', report['unique'])
        cols[1].metric('Exact novelty vs GRAMPA', f'{report["novel_unique_vs_grampa"]:.1%}')
        cols[2].metric('Natural END rate', f'{report["natural_end_rate"]:.1%}')
        history = pd.read_csv(RESULTS / 'vae_history.csv').set_index('epoch')
        st.line_chart(history[['train_reconstruction', 'validation_reconstruction']])
        st.caption('Cross-entropy per non-padding token. Lower is better; training includes word dropout and sampled latent vectors, validation uses the latent mean.')
        st.write(f'Test teacher-forced token accuracy: **{report["test_teacher_forced_token_accuracy"]:.1%}**. This is not whole-peptide reconstruction accuracy.')
        st.info(report['notes'])
        st.dataframe(pd.read_csv(RESULTS / 'vae_raw_generation.csv').head(20), hide_index=True)
        with st.expander('Complete VAE metrics'):
            st.json(report)
    with tabs[3]:
        st.write('GRAMPA supplies the generator sequences. Published UniDL4BioPep activity and toxicity datasets supply the classifier labels. External files are pinned to a Git commit with SHA-256 checksums.')
        st.markdown('[GRAMPA source](https://github.com/zswitten/Antimicrobial-Peptides) · [UniDL4BioPep source](https://github.com/dzjxzyd/UniDL4BioPep)')
        st.json(json.loads((ROOT / 'Data/external/sources.json').read_text()))
        st.warning('These are sequence-disjoint splits, not sequence-similarity-clustered splits. Benchmark performance can overestimate performance on unrelated peptide families. Activity negatives also have different length distributions; the baseline can exploit this dataset bias.')

else:
    hero('Understand every step.', 'A compact explanation for the project demonstration and viva.')
    explanations = {
        '1. What is our problem?': 'We want to narrow a large sequence search space into candidates for future experimental investigation. We combine generation with two independent screening tasks.',
        '2. How does the VAE work?': 'The encoder LSTM reads amino acids and predicts a latent mean and variance. Reparameterization samples z = mean + standard deviation × noise. The decoder LSTM predicts the next amino acid from z and previous residues. Training balances reconstruction cross-entropy and KL regularization.',
        '3. Why use KL loss and word dropout?': 'KL keeps the latent distribution near a standard normal, so random latent sampling is possible. Warmup gradually adds this constraint. Word dropout removes some previous-token information during training to encourage the decoder to use z.',
        '4. How do the predictors work?': 'Each task has its own labels and weights. A CNN detects local patterns with kernels of sizes 3, 5, and 7; global max pooling combines them with 25 simple descriptors. A random forest is an interpretable baseline. Validation AUC chooses CNN, forest, or their equal-weight ensemble.',
        '5. What is training vs validation vs test?': 'Training adjusts weights. Validation selects a checkpoint and model. The published test split is reserved for final evaluation. Identical sequences cannot appear across splits after cleaning. Similar, nonidentical sequences can still cross splits.',
        '6. What do the metrics mean?': 'Precision: how many predicted positives are correct. Recall: how many actual positives are found. F1 balances both. ROC-AUC measures ordering of positives vs negatives across thresholds. MCC measures classification agreement while accounting for all four confusion-matrix cells.',
        '7. How do we rank candidates?': 'Priority = 100 × activity score × (1 − toxicity score). A candidate needs both a high activity score and a low toxicity score to rank highly. This is a decision rule, not proof that it works or is safe.',
        '8. How is this different from UniDL4BioPep?': 'We use its published benchmark data, not its reported results or pretrained weights. Our compact CNN learns residue embeddings directly and adds interpretable features; we do not claim to reproduce its ESM-2 embedding architecture.',
        '9. What are the limitations?': 'No laboratory experiments; no MIC or bacterial-species prediction; toxicity labels are dataset-specific; similarity leakage and length bias remain possible; classifier scores are not calibrated; sequence novelty is exact-match novelty only.',
    }
    for title, body in explanations.items():
        with st.expander(title, expanded=title.startswith('1.')):
            st.write(body)
    st.subheader('Three-person presentation split')
    st.write('**Chirag:** data, VAE, and generation. **Hurshvardhan:** activity/toxicity classifiers and evaluation. **Aniket:** quality filters, ranking, interface, and interpretation.')
    st.subheader('Reproduce the complete project')
    st.code('python -m peptide_lab.training\npython -m peptide_lab.pipeline\npython -m unittest discover -s tests -v\npython -m streamlit run app.py', language='bash')
