"""Create a self-contained final report, research notebook, and evidence manifest."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import base64
import hashlib
import html
import json
import re
from datetime import datetime,timezone
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from peptide_lab.config import ROOT,RESULTS,MODELS,DATA

def cell(kind,text):
    out={'cell_type':kind,'id':hashlib.sha256(text.encode()).hexdigest()[:8],'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code':out.update(execution_count=None,outputs=[])
    return out

def main():
    folder=RESULTS/'extended'
    evidence=json.loads((folder/'research_summary.json').read_text())
    preprocessing=json.loads((RESULTS/'preprocessing_audit.json').read_text())
    study=pd.read_csv(folder/'toy_study_summary.csv')
    runs=pd.read_csv(folder/'toy_study_runs.csv')
    figures=folder/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,ax=plt.subplots(figsize=(8,4))
    d=pd.DataFrame(evidence['latent_diagnostics']).iloc[:3]
    ax.bar(['Encoder mean','Zero latent','Shuffled latent'],d.cross_entropy,color=['#16858a','#ca8a36','#be5c5c'])
    ax.set(ylabel='Validation cross-entropy per token',title='Does the decoder use the latent vector?')
    fig.tight_layout();fig.savefig(figures/'latent_ablation.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    fixed=runs[(runs.latent==24)&(runs.word_dropout==.2)]
    ax.plot(range(len(fixed)),fixed.test_cross_entropy,'o-',color='#16858a')
    ax.set(xticks=range(len(fixed)),xticklabels=fixed.seed.astype(str),ylabel='Test cross-entropy per token',xlabel='Subset / initialization seed',title='Ten-seed toy VAE robustness')
    ax.tick_params(axis='x',rotation=45)
    fig.tight_layout();fig.savefig(figures/'toy_robustness.png');plt.close(fig)
    base=(ROOT/'Documentation/FINAL_REPORT.html').read_text(encoding='utf-8')
    # Embed every plot so the report can be shared as one offline file.
    def embed(match):
        relative=match.group(1)
        image=(ROOT/'Documentation'/relative).resolve()
        return 'src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"'
    base=re.sub(r'src="(\.\./Results/[^"]+)"',embed,base)
    base=base.replace('The five pages are Overview, Generate &amp; rank, Screen sequences, Model results and Learn the project.', 'The dashboard includes Overview, Dataset explorer, Generate &amp; rank, Screen sequences, Prediction history, Model results, Research checks and Learn the project.')
    def table(frame):return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')
    def picture(name):return '<img src="data:image/png;base64,'+base64.b64encode((figures/name).read_bytes()).decode()+'">'
    extra=f'''<h2>11. Dataset prediction workspace</h2>
<p>Users can browse GRAMPA, the activity benchmark or the toxicity benchmark, filter by sequence, length, source split and known label, then score selected peptides, the first rows or a seeded random sample. Both predictions and each model's split membership are displayed. Missing labels remain missing. A label from one task is never substituted for a label from another task. Results can be inspected individually and exported; the last ten runs remain in the browser session.</p>
<h2>12. Reproducible raw-data audit</h2><p>The new preprocessing module reconstructs {preprocessing['canonical_unique_sequences']} canonical unique sequences from {preprocessing['raw_records']} records and exactly reproduces the existing sequence set and order. Length filtering yields {preprocessing['length_counts']['5_40']} at 5–40 and {preprocessing['length_counts']['5_60']} at 5–60 residues. No organism or modification filter is applied; models see the sequence only.</p>
<h2>13. Fixed-model latent diagnostics</h2>{table(pd.DataFrame(evidence['latent_diagnostics']))}
{picture('latent_ablation.png')}
<p>Using the same validation sequences and true decoder prefixes, zeroing or shuffling latent vectors increases reconstruction loss. This provides direct evidence that the current decoder uses latent information under this protocol. All {evidence['active_latent_dimensions']} dimensions exceed the stated encoder-mean variance threshold of 0.01. This does not establish biologically meaningful latent factors or eliminate every form of collapse. Free-running whole-sequence reconstruction remains weak, with about 1.1% exact matches.</p>
<h2>14. Predictor diagnostics and uncertainty</h2>{table(pd.DataFrame(evidence['predictor_checks']))}
<p class="note">Length alone achieves strong activity performance. The short activity test subset contains only eight negatives, so its very high raw accuracy is misleading; balanced accuracy is 0.625. The dataset does not establish strong discrimination among diverse short non-AMP peptides. These are post-hoc diagnostics of existing test data, not new untouched evaluations.</p>
{table(pd.DataFrame(evidence['confidence_intervals']))}
<p>Intervals use 300 stratified bootstrap resamples and quantify test-sample uncertainty only. Model scores remain uncalibrated; Brier scores and reliability-bin CSVs are reported as diagnostics. False-positive and false-negative examples are saved separately. No test-based tuning was performed.</p>
<h2>15. Controlled toy experiments</h2><p>Completed {len(runs)} new experiments, each with 200 sequences, 160/20/20 splits, 30 epochs, batch size 32, fixed maximum beta 0.02 and a ten-epoch warmup. Ten seeds assess the default latent-24, dropout-0.2 configuration. Latent dimensions 16, 32 and 64 and dropout 0.0 are tested against the default using three matched seeds. Each run retains its exact splits, configuration, weights, history, generated sequences and metrics. These are new experiments, not the missing historical Notebook 02B.</p>
{picture('toy_robustness.png')}
{table(study[study.metric.isin(['test_cross_entropy','token_accuracy'])])}
<p>The three-seed comparisons must use the matched-three-seed default row, not the ten-seed default mean. Bootstrap intervals with only three runs are highly uncertain. Differences are small and do not justify a claim that a particular latent dimension or word-dropout setting is universally optimal. The deployed full-data model was not replaced based on this study.</p>
<h2>16. Ranking sensitivity</h2>{table(pd.DataFrame(evidence['ranking_sensitivity']))}
<p>The alternative weights are illustrative preferences, not learned biological utility. See the table above for top-10 overlap on the complete saved passed-filter pool. This does not establish stability across arbitrary pools.</p>
<h2>17. Final implementation status</h2><p>The software workflow, dataset prediction interface, source-data preprocessing, model evaluation, latent diagnostic study, repeated toy runs, ranking sensitivity, executed teaching notebooks and exportable report are implemented. Exact paper reproduction, recovery of missing historical artifacts, similarity-disjoint external validation, assay-specific toxicity prediction and experimental biological validation remain outside the demonstrated claims. The original historical files and deployed models are preserved.</p>
<p>Source-paper provenance note: UniDL4BioPep identifies ATSE as the toxicity benchmark reference. The pinned repository files used here have different per-split class counts from the paper's table; this report uses actual local counts and makes no direct performance-equivalence claim. See <a href="https://www.grains.k-state.edu/research/ccl/files/publication_pdf/2023_BIB_Du_UniDL4BioPep.pdf">the authors' paper</a> and the data card for details.</p>
'''
    base=base.replace('</main>',extra+'</main>')
    (ROOT/'Documentation/COMPLETE_PROJECT_REPORT.html').write_text(base,encoding='utf-8')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':[
        cell('markdown','# Dataset prediction and research evidence\nThis notebook demonstrates the new workflow using saved models and completed evaluations. It does not retrain the production models.'),
        cell('code',"from pathlib import Path\nimport sys,json\nimport pandas as pd\nROOT=Path.cwd()\nif not (ROOT/'peptide_lab').exists(): ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nfrom peptide_lab.config import RESULTS\nfrom peptide_lab.datasets import catalog,browse,attach_labels\nfrom peptide_lab.pipeline import ScreeningPipeline"),
        cell('markdown','## 1. Raw-data preprocessing audit\nCanonical amino-acid filtering, exact deduplication and explicit length rules are reproducible from the raw file. The original empty notebook remains preserved.'),
        cell('code',"print(json.dumps(json.loads((RESULTS/'preprocessing_audit.json').read_text()),indent=2))"),
        cell('markdown','## 2. Browse labeled dataset peptides\nUse a toxicity test subset. A missing activity label must remain missing; toxicity and activity are independent tasks.'),
        cell('code',"data=catalog()\nselected=browse(data,'Toxicity benchmark','test',min_length=5,max_length=40).sample(5,random_state=42)\ndisplay(selected)"),
        cell('markdown','## 3. Predict activity and toxicity, then compare available labels\nJoin labels by sequence rather than row order because predictions are ranked.'),
        cell('code',"pipe=ScreeningPipeline()\nresult=attach_labels(pipe.predict(selected.sequence.tolist()),selected)\ndisplay(result[['sequence','activity_score','toxicity_score','toxicity_label','toxicity_matches_label','toxicity_split']])"),
        cell('markdown','## 4. Does the VAE use its latent vector?\nCompare encoder means with zeroed/shuffled representations under the same true prefixes. This tests model dependence, not biological meaning.'),
        cell('code',"research=json.loads((RESULTS/'extended/research_summary.json').read_text())\ndisplay(pd.DataFrame(research['latent_diagnostics']))"),
        cell('markdown','## 5. Predictor bias, uncertainty and ranking sensitivity\nInspect class counts and balanced accuracy. Eight short negative activity examples cannot support broad generalization claims.'),
        cell('code',"display(pd.DataFrame(research['predictor_checks']))\ndisplay(pd.DataFrame(research['confidence_intervals']))\ndisplay(pd.DataFrame(research['ranking_sensitivity']))"),
        cell('markdown','## 6. Completed controlled toy study\nTen seeds assess subset/initialization robustness; three matched seeds compare latent dimensions and word dropout. These are new experiments with weak small-sample uncertainty, not historical results or a full-data hyperparameter guarantee.'),
        cell('code',"runs=pd.read_csv(RESULTS/'extended/toy_study_runs.csv')\nprint('Completed runs:',len(runs))\ndisplay(runs)\nsummary=pd.read_csv(RESULTS/'extended/toy_study_summary.csv')\ndisplay(summary[summary.metric=='test_cross_entropy'])"),
        cell('markdown','## 7. Demonstration\nRun `python -m streamlit run app.py`. Choose Dataset explorer → Toxicity benchmark → test → Predict dataset activity & toxicity. Then inspect a peptide, export results and open Prediction history. The Research checks page explains the completed studies.\n\nReproduce additional evidence with `EVALUATE_PROJECT.bat`. Missing original historical notebooks and wet-lab validation cannot be reconstructed from model files.'),
    ]}
    (ROOT/'Notebooks/04_Dataset_Prediction_and_Research.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
    tracked=list((ROOT/'peptide_lab').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'app.py',ROOT/'requirements-lock.txt',DATA/'raw/grampa.csv']+list((DATA/'final').glob('*.csv'))+list(MODELS.glob('*'))
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':'Frozen deployed models; post-hoc diagnostics; 22 independent controlled toy runs, not selected for deployment.',
              'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.is_file()}}
    (folder/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Self-contained report, figures, notebook and evidence manifest created.')

if __name__=='__main__':main()
