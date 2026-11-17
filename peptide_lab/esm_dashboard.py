"""Evidence and architecture page for the charter's pretrained predictor route."""
import json
import pandas as pd
import streamlit as st
from .config import RESULTS, ROOT

def esm_page(pipeline):
    paths=[RESULTS/f'esm2/{task}_metrics.json' for task in ['activity','toxicity']]
    if not all(p.exists() for p in paths):
        st.info('The ESM-2 embedding/classifier comparison is being prepared. The compact prediction engine remains available.')
        return
    st.write('**Architecture:** frozen Meta ESM-2 (6 layers, 8M parameters) → mean of residue embeddings (320 features) → train-only scaling → 1D CNN → binary activity/toxicity score.')
    st.caption('CLS, EOS and padding are excluded from pooling. Heads are trained separately. This follows the pretrained-representation + CNN approach, with documented implementation differences from UniDL4BioPep.')
    rows=[]
    for task,path in zip(['activity','toxicity'],paths):
        esm=json.loads(path.read_text())
        compact=json.loads((RESULTS/f'{task}_metrics.json').read_text())
        for name,values in [('Compact baseline',compact['test'][compact['selected_model']]),('ESM-2 + CNN',esm['test'])]:
            rows.append({'task':task,'model':name,**{k:v for k,v in values.items() if k!='confusion_matrix'}})
        st.write(f'**{task.title()} validation recommendation:** {esm["validation_recommendation"]} · ESM AUC {esm["validation_auc"]:.4f} · compact AUC {esm["baseline_validation_auc"]:.4f}')
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.info('Choose the prediction engine in the sidebar to use either model throughout dataset screening, custom input and generation. Exports identify the engine used. The compact engine remains the default because it achieved higher validation AUC for both tasks.')
    tabs=st.tabs(['Activity ESM learning','Toxicity ESM learning','Reproducibility'])
    for tab,task in zip(tabs[:2],['activity','toxicity']):
        with tab:
            st.line_chart(pd.read_csv(RESULTS/f'esm2/{task}_history.csv').set_index('epoch')[['validation_auc']])
            report=json.loads((RESULTS/f'esm2/{task}_metrics.json').read_text())
            st.dataframe(pd.DataFrame(report['test']['confusion_matrix'],index=['Actual negative','Actual positive'],columns=['Predicted negative','Predicted positive']))
    with tabs[2]:
        st.json(json.loads((ROOT/'Models/esm2_backbone/source.json').read_text()))
        st.write('The downloaded backbone is stored locally for offline inference. Sequence-keyed embeddings are cached. Only training features fit the scaler and head; validation chooses checkpoints. The existing test data had already been inspected in baseline development, so this is a follow-up comparison rather than an untouched external benchmark.')
        st.code('python scripts/download_esm2.py\npython -m peptide_lab.esm_predictor')

def alignment_checks_page():
    folder=RESULTS/'alignment'
    path=folder/'similarity_summary.json'
    if path.exists():
        st.subheader('High sequence-similarity audit')
        report=json.loads(path.read_text())
        st.dataframe(pd.DataFrame(report['tasks']),hide_index=True,use_container_width=True)
        st.caption(report['method'])
        st.warning('Similarity findings diagnose possible family overlap. They do not convert the original splits into similarity-disjoint test sets.')
    composition=folder/'amino_acid_composition.csv'
    if composition.exists():
        st.subheader('Training versus generated amino-acid composition')
        st.bar_chart(pd.read_csv(composition).set_index('amino_acid'))
    losses=folder/'vae_objective_history.csv'
    if losses.exists():
        st.subheader('Weighted VAE objective')
        st.line_chart(pd.read_csv(losses).set_index('epoch')[['train_total_at_epoch_beta','validation_total_at_epoch_beta']])
        st.caption('Total = non-padding token reconstruction mean + epoch beta × per-sequence KL. Training uses word dropout and sampled z; validation uses the latent mean.')
