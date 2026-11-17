"""Build the final document-alignment addendum from measured local artifacts."""
from pathlib import Path
import hashlib
import html
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from peptide_lab.config import RESULTS

def main():
    docs=ROOT/'Documentation'
    rows=[]
    for task in ['activity','toxicity']:
        compact=json.loads((RESULTS/f'{task}_metrics.json').read_text())
        esm=json.loads((RESULTS/f'esm2/{task}_metrics.json').read_text())
        for name,metric,val in [('Compact '+compact['selected_model'],compact['test'][compact['selected_model']],compact['validation_auc'][compact['selected_model']]),('ESM-2 + CNN',esm['test'],esm['validation_auc'])]:
            rows.append({'task':task,'model':name,'validation_auc':val,**{k:metric[k] for k in ['accuracy','precision','recall','f1','roc_auc','mcc']}})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(RESULTS/'alignment/final_model_comparison.csv',index=False)
    audit=json.loads((RESULTS/'generation_audit.json').read_text())
    thresholds=json.loads((RESULTS/'alignment/property_thresholds.json').read_text())
    matrix='''| Requirement from supplied documents | Final implementation / evidence |
|---|---|
| LSTM VAE generation and latent sampling | Saved full-data VAE, KL warmup, reconstruction/KL/total-objective curves and latent controls |
| GRAMPA preparation and experimental context | Raw CSV retained with assay/context columns; canonical sequence training views are separate. No incompatible MIC values are pooled into labels |
| Small preliminary experiment | 22 completed controlled runs; ten default seeds and three matched seeds for architecture/dropout comparisons |
| Pretrained representation plus CNN | Frozen official ESM-2 t6 8M backbone, residue mean pooling, train-only scaler, two separately trained CNN heads; selectable throughout dashboard |
| Independent activity and toxicity tasks | Separate labeled datasets, splits, heads and metrics; labels joined by sequence only when available |
| Reconstruction, validity, diversity and novelty | Raw sampling statistics, sequence-length plots, amino-acid composition, teacher-forced accuracy and free-running diagnostics |
| Duplicate and similarity checks | Exact-disjoint task splits plus exhaustive normalized-edit-similarity audit; no claim of homology-disjoint evaluation |
| Sequence filters and property screen | Sequence rejection audit plus validation-selected activity/toxicity cutoffs; all sequence survivors retained and property flags exported |
| Weighted activity-minus-toxicity ranking | Selectable weighted difference with explicit weight and normalized display; original product rule also retained |
| Classification evaluation | Held-out accuracy, precision, recall, F1, MCC, ROC-AUC, confusion matrices, validation learning curves; bootstrap diagnostics for compact models |
| Showable final application | Nine Streamlit pages; dataset/custom/generated predictions, history, CSVs, measured evidence and teaching notebooks |
'''
    text='''# Final alignment with the project charter, literature review and presentation

This document reconciles SLgroup_04.pdf, Statistical_Learning.pdf and AMP_Statistical_Learning_Project_Presentation_Final.pptx. The PDFs and presentation are reference material, not evidence that an experiment has been run. The implemented computational scope is complete; biological efficacy and safety are not established.

'''+matrix+'''
## Architecture and learning workflow

GRAMPA raw records -> canonical unique sequences -> train/validation/test split -> LSTM encoder -> mean/log-variance -> sampled latent vector -> LSTM decoder -> sequence filters -> two independent predictors -> property-screen flags -> ranking -> dashboard and complete CSV.

The compact activity predictor is the validation-selected average of a sequence CNN and random forest. Toxicity uses its independently selected CNN. Each CNN learns residue embeddings and combines sequence features with 25 descriptors. The alternative ESM-2 route freezes a pretrained six-layer, 8M-parameter protein transformer, averages non-special residue embeddings to 320 values, applies a training-only StandardScaler and a separate Conv1D classifier per task. That convolution operates along pooled embedding features, not amino-acid positions; this is an explicit architectural adaptation, not exact UniDL4BioPep reproduction. No task labels update the backbone. Cached embeddings do not fit a model on held-out labels.

Both tasks' validation AUC favors the compact models, so the compact engine remains the default. ESM-2 is fully implemented, locally stored and selectable. Larger pretrained models are not assumed to outperform a measured baseline. Model selection and cutoff selection use validation data; the new ESM comparison is a follow-up using already inspected benchmark test data, not fresh external validation.

## Measured results

'''+comparison.round(6).to_csv(index=False)+f'''
The saved demonstration samples {audit['sampled']} sequences, retains {audit['passed_filters']} after sequence filters and flags {audit['passed_property_screen']} as passing the additional property screen. All {audit['returned']} sequence-filter survivors receive ranks and remain in plots and CSVs. High scores and passing flags are model outputs, not confirmed active/non-toxic peptides.

## Property screen versus class prediction

Class predictions and reported benchmark accuracy use 0.5. The additional property screen uses validation-selected thresholds: compact activity >= {thresholds['compact']['activity']['threshold']}, toxicity < {thresholds['compact']['toxicity']['threshold']}; ESM activity >= {thresholds['esm2']['activity']['threshold']}, toxicity < {thresholds['esm2']['toxicity']['threshold']}. Activity maximizes validation F1 over the prespecified 0.01..0.99 grid; toxicity takes the highest cutoff reaching at least 95% validation recall for toxic examples. These follow-up rules do not guarantee equivalent recall on new data. Exact validation-file hashes, rules and scores are retained in Results/final/alignment/property_thresholds.json.

Product ranking is 100*A*(1-T). The document equation is also available: S=w*A-(1-w)*T. Displayed priority is 100*(S+1-w), preserving order while mapping the possible range to 0..100. Weights are user preferences chosen before a run, not fitted biological utility. Exports preserve the actual engine, rule, weight, raw weighted score, cutoffs and flags. History preserves earlier results when sidebar settings change.

## Historical presentation results

The fifth physical slide (Current Results) reports an earlier 41-epoch run, best validation loss 1.9421 at epoch 31, 50 valid samples with five unique and two novel sequences, 0% exact reconstruction, 46.7% token accuracy and KL 3.58e-6. Those values are historical claims in the supplied slide, not measurements recovered from available original checkpoints. They must not be silently relabeled as current results.

The delivered full-data run uses 30 epochs. Current held-out teacher-forced token accuracy is 48.95% and reconstruction cross-entropy is 1.6697. A separate raw 500-sample evaluation produced 500 unique sequences with no exact GRAMPA matches and 100% natural termination. Decoding enforces alphabet/minimum-length rules. These samples differ from the 750-sample filtered demonstration; their counts must not be mixed. Teacher-forced reconstruction, free-running exact reconstruction, classification accuracy and sequence validity are different measurements.

## Scientific boundaries to explain honestly

The supplied Statistical_Learning.pdf is a literature review/proposed framework, not an additional course grading rubric. Referenced models are background, not a requirement to implement every paper. Binary activity and dataset-specific toxicity are implemented. Species-specific MIC regression, assay-specific hemolysis, laboratory validation, exact reproduction of all cited papers and new discoveries are not claimed.

Exact duplicates are separated within each task, but normalized edit similarity >=0.8 occurs for 943/8877 activity test sequences, 118/580 toxicity sequences and 421/651 VAE sequences versus their training sets. This is not BLAST identity or evolutionary homology. The final project measures the overlap rather than falsely claiming a clustered split. Activity has strong length bias; the short-peptide test subset has only eight negatives and balanced accuracy 0.625. Benchmark accuracy must not be generalized to all novel peptides.

## Reproduce and demonstrate

Install requirements-lock.txt and run RUN_PROJECT.bat (or python -m streamlit run app.py). Saved models run locally without downloading weights. Start with Dataset explorer -> Toxicity benchmark -> test -> random sample -> Predict. Explain available labels; then switch to ESM-2 and run a small batch. Generate 500 raw sequences, explain the sequence-filter count and separate property flag, hover over ranks, and download all rows. Show Model results, ESM-2 comparison, Research checks, and Prediction history.

For ESM retraining: python -m peptide_lab.esm_predictor. For cutoff regeneration: python -m peptide_lab.screening_thresholds. For similarity/composition evidence: python -m peptide_lab.alignment_checks. Run python scripts/verify_project.py for software checks. The supplied weights already work; retraining is optional. Rebuild reports after any model changes. See FINALIZE_PROJECT.bat for the ordered evidence/report workflow without retraining production weights.

Start your explanation with: "We learn sequence patterns using a VAE, score generated or existing peptides using two independent supervised tasks, and prioritize candidates for further investigation. We validate software and benchmark performance, not biological efficacy."
'''
    (docs/'FINAL_ALIGNMENT_AND_TEAM_GUIDE.md').write_text(text,encoding='utf-8')
    slides='''# Presentation update guide

Use this with the eight physical slides of the supplied presentation. Preserve the original as historical material; update your presentation copy before presenting current results.

1. **Title:** PeptideLab: Antimicrobial Peptide Generation and Computational Screening. Include team/course details.
2. **Motivation:** explain why generation needs both activity and toxicity screening; acknowledge class imbalance and sequence similarity.
3. **VAE:** show encoder -> mean/log variance -> reparameterization -> decoder, reconstruction + beta*KL. Explain word dropout and validation selection in plain language.
4. **Data:** GRAMPA trains generation; activity and toxicity have separate labels. Show preprocessing and exact-disjoint splits. Preserve raw experimental context; do not imply common MIC conditions.
5. **Current results:** replace the earlier 41-epoch collapse figures with the measured current results in FINAL_ALIGNMENT_AND_TEAM_GUIDE.md. Label the earlier figures as historical only if you retain them. Show both the 500 raw-sample evaluation and 750-sample filter funnel with distinct labels. Never call unique sequences experimentally validated AMPs.
6. **Predictors:** show both compact and frozen ESM-2 + CNN routes. Present the actual comparison table in Results/final/alignment/final_model_comparison.csv. Compact wins validation selection for both tasks; ESM remains an implemented alternative.
7. **Screen/rank:** explain sequence filters, validation-selected property flags, product and weighted-difference ranking. All sequence survivors are exported; the property flag defines an additional subset. Weights express preferences, not probabilities.
8. **Conclusion/demo:** run a dataset batch, show known labels, generate/rank, export CSV and show held-out results. End with length bias, sequence-similarity overlap, uncalibrated scores and need for laboratory validation.

The original PPT is not automatically overwritten. This guide provides the reconciled narrative for the final delivered code and measured results.
'''
    (docs/'PRESENTATION_UPDATE_GUIDE.md').write_text(slides,encoding='utf-8')
    report=docs/'COMPLETE_PROJECT_REPORT.html'
    content=report.read_text(encoding='utf-8')
    marker='<!-- FINAL_ALIGNMENT -->'
    if marker in content:content=content.split(marker)[0]+'</main></body></html>'
    addition=marker+'<h2>18. Final three-document alignment and measured ESM-2 comparison</h2>'+comparison.round(6).to_html(index=False)+'<pre style="white-space:pre-wrap;font:inherit">'+html.escape(text)+'</pre>'
    report.write_text(content.replace('</main>',addition+'</main>'),encoding='utf-8')
    def cell(kind,source):
        c={'cell_type':kind,'metadata':{},'source':source.splitlines(keepends=True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        return c
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        cell('markdown','# Final project alignment and model comparison\nRead the team guide for architecture and limitations. This notebook uses saved evidence and weights; it does not retrain models.'),
        cell('code',"from pathlib import Path\nimport sys,json\nimport pandas as pd\nROOT=Path.cwd()\nif not (ROOT/'peptide_lab').exists(): ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nfrom peptide_lab.config import RESULTS\nfrom peptide_lab.pipeline import ScreeningPipeline\ndisplay(pd.read_csv(RESULTS/'alignment/final_model_comparison.csv'))"),
        cell('code',"print(json.dumps(json.loads((RESULTS/'alignment/property_thresholds.json').read_text()),indent=2))\nprint(json.dumps(json.loads((RESULTS/'generation_audit.json').read_text()),indent=2))"),
        cell('markdown','## Both engines can screen the same existing sequences\nLabels belong to separate tasks; scores are not calibrated safety probabilities.'),
        cell('code',"sequences=['GILSGILGAGKSLVCGLSGLC','KWKLFKKIGAVLKVL']\nfor engine in ['compact','esm2']:\n    pipe=ScreeningPipeline(engine,ranking_rule='weighted_difference')\n    display(pipe.predict(sequences)[['rank','sequence','prediction_engine','activity_score','toxicity_score','priority_score','passes_property_screen']])"),
        cell('code',"display(pd.DataFrame(json.loads((RESULTS/'alignment/similarity_summary.json').read_text())['tasks']))\ndisplay(pd.read_csv(RESULTS/'alignment/amino_acid_composition.csv'))")
    ]}
    (ROOT/'Notebooks/05_Final_Alignment_and_ESM2.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print('Final alignment guide, slide update guide, report addendum and notebook created.')

if __name__=='__main__':main()
