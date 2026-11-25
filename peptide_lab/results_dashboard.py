"""Display saved evaluation results independently of the engine used for predictions."""
import json
import pandas as pd
import streamlit as st
from .config import RESULTS
from .tuning_dashboard import tuning_page


COMPACT = 'Compact baseline'
ESM = 'ESM-2 + CNN'
COMPARISON = 'Engine comparison'
VAE = 'VAE generator'
TASKS = ('activity', 'toxicity')


def classifier_results(task, section):
    folder = RESULTS / 'esm2' if section == ESM else RESULTS
    path = folder / f'{task}_metrics.json'
    if not path.exists():
        st.info(f'The {section} evaluation for {task} is unavailable.')
        return
    report = json.loads(path.read_text(encoding='utf-8'))
    if section == COMPACT:
        model = report['selected_model']
        models = report['test']
        st.write(f'Validation selected **{model}** for general {task}.')
    else:
        model = ESM
        models = {model: report['test']}
        st.write(f'Frozen ESM-2 embeddings feed a CNN trained on {task} labels.')
    rows = [{'Model': name, **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}}
            for name, metrics in models.items()]
    labels = {'accuracy':'Accuracy', 'precision':'Precision', 'recall':'Recall', 'f1':'F1',
              'mcc':'MCC', 'roc_auc':'ROC-AUC', 'average_precision':'Average precision'}
    st.dataframe(pd.DataFrame(rows).rename(columns=labels), hide_index=True, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown('**CNN validation ROC-AUC by epoch**')
        st.line_chart(pd.read_csv(folder / f'{task}_history.csv').set_index('epoch')[['validation_auc']])
    with right:
        st.markdown(f'**Test confusion matrix · {model}**')
        st.dataframe(pd.DataFrame(models[model]['confusion_matrix'], index=['Actual negative','Actual positive'],
                                  columns=['Predicted negative','Predicted positive']))
    if section == COMPACT:
        with st.expander('Random forest feature importance'):
            importance = pd.read_csv(RESULTS / f'{task}_feature_importance.csv').head(10)
            st.bar_chart(importance.set_index('feature'))


def comparison():
    st.subheader('General activity and toxicity')
    rows = []
    for task in TASKS:
        compact = json.loads((RESULTS / f'{task}_metrics.json').read_text(encoding='utf-8'))
        path = RESULTS / 'esm2' / f'{task}_metrics.json'
        if not path.exists():
            st.info(f'The ESM-2 evaluation for {task} is unavailable.')
            continue
        esm = json.loads(path.read_text(encoding='utf-8'))
        name = compact['selected_model']
        for label, validation, test in [
            (f'{COMPACT} ({name})', compact['validation_auc'][name], compact['test'][name]),
            (ESM, esm['validation_auc'], esm['test']),
        ]:
            rows.append({'Task': task.capitalize(), 'Engine': label, 'Validation ROC-AUC': validation,
                         'Test accuracy': test['accuracy'], 'Test ROC-AUC': test['roc_auc'], 'Test F1': test['f1']})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption('Within each task, both engines use the same held-out test peptides. Related peptide families can occur across splits.')

    path = RESULTS / 'target_benchmark/comparison.csv'
    if path.exists():
        st.subheader('Organism-specific activity')
        cols = {'target':'Organism', 'test_rows':'Test peptides', 'accuracy':'Compact accuracy',
                'esm2_accuracy':'ESM-2 accuracy', 'auc':'Compact ROC-AUC', 'esm2_auc':'ESM-2 ROC-AUC'}
        values = pd.read_csv(path)
        st.dataframe(values.filter(items=cols).rename(columns=cols), hide_index=True, use_container_width=True)


def results_page():
    section = st.radio('Results section', [COMPACT, ESM, COMPARISON, VAE], horizontal=True, key='results_section')
    if section == COMPARISON:
        comparison()
        return
    if section == VAE:
        view = st.radio('Results view', ['Baseline results','Hyperparameter tuning'], horizontal=True, key='results_view_vae')
        if view == 'Hyperparameter tuning':
            tuning_page('vae')
            return
        report = json.loads((RESULTS / 'vae_metrics.json').read_text(encoding='utf-8'))
        cols = st.columns(3)
        cols[0].metric('Unique samples / 500', report['unique'])
        cols[1].metric('Exact novelty vs GRAMPA', f"{report['novel_unique_vs_grampa']:.1%}")
        cols[2].metric('Natural END rate', f"{report['natural_end_rate']:.1%}")
        st.line_chart(pd.read_csv(RESULTS / 'vae_history.csv').set_index('epoch')[['train_reconstruction','validation_reconstruction']])
        st.caption('Reconstruction loss per non-padding token. Lower is better.')
        st.write(f"Test teacher-forced token accuracy was **{report['test_teacher_forced_token_accuracy']:.1%}**.")
        return
    for tab, task in zip(st.tabs(['Activity predictor','Toxicity predictor']), TASKS):
        with tab:
            if section == COMPACT:
                view = st.radio('Results view', ['Baseline results','Hyperparameter tuning'], horizontal=True,
                                key='results_view_'+task)
                if view == 'Hyperparameter tuning':
                    tuning_page(task)
                    continue
            classifier_results(task, section)
