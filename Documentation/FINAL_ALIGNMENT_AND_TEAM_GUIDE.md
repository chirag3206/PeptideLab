# Final alignment with the project charter, literature review and presentation

This document reconciles SLgroup_04.pdf, Statistical_Learning.pdf and AMP_Statistical_Learning_Project_Presentation_Final.pptx. The PDFs and presentation are reference material, not evidence that an experiment has been run. The implemented computational scope is complete; biological efficacy and safety are not established.

| Requirement from supplied documents | Final implementation / evidence |
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

## Architecture and learning workflow

GRAMPA raw records -> canonical unique sequences -> train/validation/test split -> LSTM encoder -> mean/log-variance -> sampled latent vector -> LSTM decoder -> sequence filters -> two independent predictors -> property-screen flags -> ranking -> dashboard and complete CSV.

The compact activity predictor is the validation-selected average of a sequence CNN and random forest. Toxicity uses its independently selected CNN. Each CNN learns residue embeddings and combines sequence features with 25 descriptors. The alternative ESM-2 route freezes a pretrained six-layer, 8M-parameter protein transformer, averages non-special residue embeddings to 320 values, applies a training-only StandardScaler and a separate Conv1D classifier per task. That convolution operates along pooled embedding features, not amino-acid positions; this is an explicit architectural adaptation, not exact UniDL4BioPep reproduction. No task labels update the backbone. Cached embeddings do not fit a model on held-out labels.

Both tasks' validation AUC favors the compact models, so the compact engine remains the default. ESM-2 is fully implemented, locally stored and selectable. Larger pretrained models are not assumed to outperform a measured baseline. Model selection and cutoff selection use validation data; the new ESM comparison is a follow-up using already inspected benchmark test data, not fresh external validation.

## Measured results

task,model,validation_auc,accuracy,precision,recall,f1,roc_auc,mcc
activity,Compact ensemble,0.994395,0.961924,0.945892,0.920796,0.933175,0.991557,0.90672
activity,ESM-2 + CNN,0.992945,0.951786,0.883028,0.960203,0.92,0.990339,0.887174
toxicity,Compact cnn,0.968533,0.922414,0.9375,0.892193,0.914286,0.971802,0.844362
toxicity,ESM-2 + CNN,0.957356,0.898276,0.883212,0.899628,0.891344,0.966877,0.795855

The saved demonstration samples 750 sequences, retains 637 after sequence filters and flags 436 as passing the additional property screen. All 637 sequence-filter survivors receive ranks and remain in plots and CSVs. High scores and passing flags are model outputs, not confirmed active/non-toxic peptides.

## Property screen versus class prediction

Class predictions and reported benchmark accuracy use 0.5. The additional property screen uses validation-selected thresholds: compact activity >= 0.49, toxicity < 0.08; ESM activity >= 0.85, toxicity < 0.14. Activity maximizes validation F1 over the prespecified 0.01..0.99 grid; toxicity takes the highest cutoff reaching at least 95% validation recall for toxic examples. These follow-up rules do not guarantee equivalent recall on new data. Exact validation-file hashes, rules and scores are retained in Results/final/alignment/property_thresholds.json.

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
