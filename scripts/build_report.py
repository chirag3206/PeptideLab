"""Build an offline report and teaching notebook from actual saved evaluation artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import html
import json
import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve
from peptide_lab.config import ROOT, RESULTS, MODELS

def read(name):
    return json.loads((RESULTS / name).read_text())

def main():
    figures = RESULTS / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11, 'figure.dpi': 140})
    summaries, audit_tables = [], []
    reports = {}
    for task in ['activity', 'toxicity']:
        report = read(task + '_metrics.json')
        reports[task] = report
        selected = report['selected_model']
        for model, values in report['test'].items():
            summaries.append({'Task': task.title(), 'Model': model + (' (selected)' if model == selected else ''), **{k: round(v, 4) for k, v in values.items() if k != 'confusion_matrix'}})
        audit = read(task + '_data_audit.json')
        for split, counts in audit['splits'].items():
            audit_tables.append({'Task': task, 'Split': split, **counts})
        pred = pd.read_csv(RESULTS / f'{task}_test_predictions.csv')
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for model in report['test']:
            score = pred[model + '_score']
            fpr, tpr, _ = roc_curve(pred.label, score)
            precision, recall, _ = precision_recall_curve(pred.label, score)
            ax[0].plot(fpr, tpr, label=f'{model}: {report["test"][model]["roc_auc"]:.3f}')
            ax[1].plot(recall, precision, label=model)
        ax[0].plot([0, 1], [0, 1], '--', color='gray')
        ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title=task.title() + ' · test ROC')
        ax[1].set(xlabel='Recall', ylabel='Precision', title=task.title() + ' · test precision–recall')
        for a in ax:
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f'{task}_curves.png')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(4.7, 4))
        cm = np.array(report['test'][selected]['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else '#153040', fontsize=18)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], xlabel='Predicted label', ylabel='True label', title=task.title() + ' · selected model')
        fig.tight_layout()
        fig.savefig(figures / f'{task}_confusion.png')
        plt.close(fig)
    history = pd.read_csv(RESULTS / 'vae_history.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_reconstruction, label='Train (word dropout)')
    axes[0].plot(history.epoch, history.validation_reconstruction, label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy / token', title='VAE reconstruction')
    axes[0].legend()
    axes[1].plot(history.epoch, history.validation_kl, color='#167c80')
    axes[1].set(xlabel='Epoch', ylabel='KL divergence / sequence', title='VAE validation KL')
    fig.tight_layout()
    fig.savefig(figures / 'vae_training.png')
    plt.close(fig)
    raw = pd.read_csv(RESULTS / 'vae_raw_generation.csv')
    original = pd.read_csv(ROOT / 'Data/final/vae_train.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(original.sequence.str.len(), bins=range(5, 65, 3), density=True, alpha=.5, label='Training')
    ax.hist(raw.sequence.str.len(), bins=range(5, 65, 3), density=True, alpha=.5, label='Generated (unfiltered)')
    ax.set(xlabel='Residues', ylabel='Density', title='Sequence length distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'length_distribution.png')
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / 'metrics_summary.csv', index=False)
    vae = read('vae_metrics.json')
    generation = read('generation_audit.json')
    candidates = pd.read_csv(RESULTS / 'ranked_candidates.csv')
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'streamlit', 'matplotlib', 'openpyxl', 'joblib']}
    (ROOT / 'requirements-lock.txt').write_text('\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(), 'seed': 42, 'packages': versions,
                'model_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.is_file()}}
    (RESULTS / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PeptideLab — Final Project Report</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#163247;background:#edf3f7;margin:0}}main{{max-width:1050px;margin:32px auto;background:white;padding:48px;border-radius:16px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:40px;color:#10767b}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dce6ec}}th{{background:#e5f2f1}}img{{max-width:100%;height:auto}}.hero{{background:#12364a;color:white;padding:30px;border-radius:12px}}.note{{background:#fff3dc;padding:18px;border-left:4px solid #bd8d28}}code{{background:#eef3f6;padding:3px}}@media print{{body{{background:white}}main{{margin:0;padding:10px}}h2,table,img{{break-inside:avoid}}.hero{{color:#12364a;background:white}}}}</style></head><body><main>
<div class="hero"><p>DSC4804 · STATISTICAL LEARNING · GROUP 4</p><h1>AI-Based Generation and Screening of Antimicrobial Peptides Using Deep Learning</h1><p>Chirag · Hurshvardhan Rao · Aniket Mishra</p></div>
<h2>1. Abstract</h2><p>PeptideLab implements an end-to-end computational workflow for peptide generation, activity classification, toxicity classification and candidate ranking. An LSTM variational autoencoder learns from GRAMPA sequences. Independent compact CNN predictors and random-forest baselines learn from published UniDL4BioPep labels. A local Streamlit interface provides reproducible sampling, user-input screening, evaluation views and CSV export. All results below are computed from saved evaluation artifacts, not copied from research papers.</p>
<h2>2. Problem and scope</h2><p>The objective is to reduce a large sequence search space to a computational shortlist. This implementation predicts binary antimicrobial-class membership and a benchmark toxicity label; it does not predict MIC, organism-specific potency or comprehensive safety. Candidate predictions require future experimental investigation.</p>
<h2>3. Data and preprocessing</h2><p>GRAMPA contains 51,345 experimental records and the supplied cleaned file contains 6,760 unique sequences. The final VAE uses 6,510 canonical sequences of length 5–60: {vae['splits']['train']} training, {vae['splits']['validation']} validation and {vae['splits']['test']} test sequences. Classifiers use canonical sequences of length 5–200. Published classifier test splits are retained. Conflicting-label sequences and exact duplicates are removed; any exact train/test overlap is removed from training. Twenty percent of the remaining training set is reserved for validation. Scaling is fitted only on training data.</p>
{pd.DataFrame(audit_tables).to_html(index=False, border=0)}
<p>Original downloads are pinned to a repository commit and checksummed in <code>Data/external/sources.json</code>. Detailed exclusion counts are stored in each task's data audit JSON.</p>
<h2>4. Architecture</h2><p><b>Generator:</b> embedding dimension 32; LSTM hidden dimension 96; latent dimension 24. The encoder predicts mean and log variance. Reparameterization samples a latent vector; the decoder receives this vector at every step and through its initial state. Training uses masked reconstruction cross-entropy, KL warmup to weight 0.02, word dropout 0.2, gradient clipping and validation-based checkpoint selection.</p>
<p><b>Predictors:</b> each task has a separate model. A 24-dimensional residue embedding feeds 32 convolution filters at each kernel width 3, 5 and 7. Masked global max pooling produces 96 features, concatenated with 25 standardized sequence descriptors. A 48-unit hidden layer and dropout feed a binary output. Random forests with 240 trees provide baselines. Validation ROC-AUC selects the best CNN checkpoint and one of CNN, forest or their equal-weight average.</p>
<p><b>Ranking:</b> <code>100 × activity_score × (1 − toxicity_score)</code>. This is a transparent prioritization rule, not a probability of success. Sampling filters reject truncation, out-of-range length, very low complexity, exact known sequences and duplicates.</p>
<h2>5. Held-out classification results</h2>{summary.to_html(index=False, border=0)}
<p>All metrics use the held-out cleaned published test splits. Classification threshold is 0.5. Positive means antimicrobial for activity and toxic for toxicity. Selection was performed using validation AUC, before calculating test results.</p>
<img src="../Results/final/figures/activity_curves.png" alt="Activity ROC and precision recall curves"><img src="../Results/final/figures/toxicity_curves.png" alt="Toxicity ROC and precision recall curves">
<img style="width:47%" src="../Results/final/figures/activity_confusion.png" alt="Activity confusion matrix"><img style="width:47%" src="../Results/final/figures/toxicity_confusion.png" alt="Toxicity confusion matrix">
<h2>6. Generation evaluation</h2><p>In a fixed unfiltered sample of {vae['generated']} outputs, {vae['unique']} were distinct; exact novelty against GRAMPA was {vae['novel_unique_vs_grampa']:.1%}; natural END rate was {vae['natural_end_rate']:.1%}. Teacher-forced test token accuracy was {vae['test_teacher_forced_token_accuracy']:.1%}, with cross-entropy {vae['test_teacher_forced_cross_entropy']:.3f}. These reconstruction metrics supply the true previous tokens and are not free-running reconstruction accuracy.</p>
<p class="note">Alphabet and minimum-length validity are enforced by decoding. Uniqueness and exact novelty do not demonstrate antimicrobial function or low toxicity. The generator is a statistical proposal mechanism, and quality must not be inferred solely from a large number of unique strings.</p>
<img src="../Results/final/figures/vae_training.png" alt="VAE training curves"><img src="../Results/final/figures/length_distribution.png" alt="Training versus generated length distribution">
<h2>7. End-to-end demonstration</h2><p>The saved run uses seed 42 and temperature 0.9. It samples {generation['sampled']} sequences, accepts {generation['accepted_before_limit']} before the requested limit, and returns {generation['returned']} candidates. These are the first accepted samples, subsequently ranked by the two predictors.</p>
{candidates[['sequence', 'activity_score', 'toxicity_score', 'priority_score']].head(10).round(4).to_html(index=False, border=0)}
<h2>8. Interface and reproducibility</h2><p>Run <code>python -m streamlit run app.py</code> or double-click <code>RUN_PROJECT.bat</code>. Saved weights support offline CPU inference. The five pages are Overview, Generate &amp; rank, Screen sequences, Model results and Learn the project. Retraining uses <code>python -m peptide_lab.training</code>; batch generation uses <code>python -m peptide_lab.pipeline</code>. Rebuild this report after changing models. Package versions and artifact hashes are recorded in <code>Results/final/run_manifest.json</code>.</p>
<h2>9. Limitations and future work</h2><p>There is no laboratory validation. Model scores are not calibrated probabilities. The classifier evaluation removes exact overlap but does not cluster by sequence similarity, so homologous sequences can cross splits. Activity positives and negatives have different sequence-length distributions, creating an exploitable benchmark bias. Toxicity classes are dataset-specific. Novelty checks exact matches against supplied sequence pools only. The VAE does not guarantee biological function. Further work should include similarity-clustered evaluation, score calibration, external datasets, MIC/species-specific prediction and experimental investigation.</p>
<p>The implementation uses compact learned residue embeddings, not ESM-2. It is inspired by the charter's baseline studies and uses their publicly shared benchmark data, but does not claim to reproduce their architectures or published results.</p>
<h2>10. Sources and contribution disclosure</h2><ul><li><a href="https://github.com/zswitten/Antimicrobial-Peptides">GRAMPA dataset</a></li><li><a href="https://github.com/dzjxzyd/UniDL4BioPep">UniDL4BioPep benchmark repository</a></li><li><a href="https://academic.oup.com/bib/article/24/3/bbad135/7107929">UniDL4BioPep paper</a></li><li><a href="https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.725727/full">PepVAE paper</a></li></ul><p>AI assisted implementation, testing and documentation. The team is responsible for understanding, verifying and presenting the work, and should disclose assistance according to the project charter.</p>
</main></body></html>'''
    (ROOT / 'Documentation/FINAL_REPORT.html').write_text(body, encoding='utf-8')
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}}
    def md(text):
        return {"cell_type": "markdown", "id": hashlib.md5(text.encode()).hexdigest()[:8], "metadata": {}, "source": text.splitlines(keepends=True)}
    def code(text):
        return {"cell_type": "code", "id": hashlib.md5(text.encode()).hexdigest()[:8], "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}
    nb['cells'] = [
        md('# PeptideLab — complete project walkthrough\nRun all cells to inspect the trained project. Retraining is optional and kept separate so the demonstration is fast. Read the cell explanations before your viva.'),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd()\nif not (ROOT / 'peptide_lab').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom peptide_lab.config import DATA, RESULTS\nfrom peptide_lab.pipeline import ScreeningPipeline\nprint('Project:', ROOT)"),
        md('## 1. Inspect the data\nGRAMPA trains the generator. Separate labeled benchmark datasets train activity and toxicity prediction. Exact splits and cleaning decisions are saved.'),
        code("for task in ['activity', 'toxicity']:\n    print(task.upper())\n    audit = json.loads((RESULTS / f'{task}_data_audit.json').read_text())\n    display(pd.DataFrame(audit['splits']).T)\n    print('Train/test exact overlap removed:', audit['train_test_overlap_removed_from_train'])"),
        md('## 2. Understand the neural networks\nThe LSTM VAE compresses sequences into a distribution and decodes samples. Two separate CNNs recognize activity and toxicity patterns. The forest is a baseline using simple descriptors.'),
        code("from peptide_lab.networks import PeptideVAE, PeptideCNN\nprint(PeptideVAE())\nprint(PeptideCNN())"),
        md('## 3. Read actual held-out results\nThe test set was not used for model selection. CNN/forest/ensemble selection uses validation ROC-AUC. High benchmark results do not establish biological function.'),
        code("display(pd.read_csv(RESULTS / 'metrics_summary.csv'))"),
        md('## 4. Inspect VAE learning\nCross-entropy is measured per non-padding token. Validation has no word dropout and uses the latent mean. The nonzero KL term does not by itself prove that every latent dimension is useful.'),
        code("history = pd.read_csv(RESULTS / 'vae_history.csv')\nhistory.plot(x='epoch', y=['train_reconstruction', 'validation_reconstruction'], figsize=(9,4))\nplt.ylabel('Cross-entropy per token')\nplt.show()\nprint(json.dumps(json.loads((RESULTS / 'vae_metrics.json').read_text()), indent=2))"),
        md('## 5. Screen input sequences\nThese are demonstration inputs, not newly discovered peptides. Scores predict dataset labels, not MIC or comprehensive safety.'),
        code("pipeline = ScreeningPipeline()\npredictions = pipeline.predict(['GILSGILGAGKSLVCGLSGLC', 'KWKLFKKIGAVLKVL'])\ndisplay(predictions)"),
        md('## 6. Run the full generation and ranking workflow\nSamples come from the VAE normal prior. Quality filters remove truncated, repetitive, known and duplicate sequences. The first accepted candidates are scored and ranked.'),
        code("candidates, audit = pipeline.generate(count=10, seed=42)\ndisplay(candidates[['sequence','activity_score','toxicity_score','priority_score','length']])\nprint(json.dumps(audit, indent=2))"),
        md('## 7. Explain the ranking\n`priority = 100 × activity_score × (1 − toxicity_score)`. This explicit preference score is not a calibrated probability of safety or success.'),
        code("import numpy as np\nassert np.allclose(candidates.priority_score, 100*candidates.activity_score*(1-candidates.toxicity_score))\nprint('Ranking formula verified.')\nprint('Exact GRAMPA matches:', int(candidates.seen_in_grampa.sum()))"),
        md('## 8. Reproduce training (optional)\nRun `python -m peptide_lab.training` from the project root. Then run `python -m peptide_lab.pipeline` and `python scripts/build_report.py` to refresh the demonstration and report. Do not retrain during the live evaluation.\n\n## 9. Limitations\nNo experimental validation; no MIC/species model; uncalibrated scores; exact-match rather than similarity-disjoint splits; length bias; exact novelty only. We use learned residue embeddings, not the baseline paper’s ESM-2 architecture.\n\n## 10. Show the frontend\nRun `python -m streamlit run app.py` and open http://127.0.0.1:8501.'),
    ]
    nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': platform.python_version()}}
    (ROOT / 'Notebooks/03_Final_Project_Walkthrough.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print('Report, figures, walkthrough notebook, dependency lock and manifest created.')

if __name__ == '__main__':
    main()
