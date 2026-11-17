# PeptideLab: demonstration and viva guide

## Tonight's preparation

1. Each member reads this guide and the app's Learn the project page.
2. Run the app once, generate a batch, and download a CSV.
3. Read the actual Model results page; do not memorize numbers from the baseline papers.
4. Keep the entire project folder together. Saved models, data splits, and reports are required.
5. Do not start retraining immediately before evaluation. Use the saved model artifacts.

## A seven-minute demonstration

**New dataset demonstration:** Open Dataset explorer, choose Toxicity benchmark and the test split, leave the three selected peptides, and click Predict dataset activity & toxicity. Inspect one row: explain its predicted score, known label, and dataset split. Switch to Activity benchmark for activity labels, or GRAMPA to explore generator data. Unknown labels remain blank. Prediction history lets you revisit and export previous runs. Research checks shows the newly completed evidence; use the latest `COMPLETE_PROJECT_REPORT.html` and notebook 04 for these additions.

**0:00–0:45 — Problem.** “Our project is a computational peptide screening prototype. We generate sequences, estimate antimicrobial-class and toxic-class scores, and rank candidates for future investigation.”

**0:45–2:00 — Generation.** Open Generate & rank. Use 30 candidates, seed 42, temperature 0.9, length 10–40. Explain that seed controls repeatability and temperature controls sampling diversity. Run the pipeline. The VAE samples a latent vector and decodes one residue at a time; the app is not showing a hardcoded list.

**2:00–3:00 — Screening and ranking.** Explain the two scores and the priority formula. The top row has a favorable combination according to the models, not verified effectiveness or safety. Show novelty flags and download the CSV.

**3:00–4:00 — User input.** Open Screen sequences. Run the provided examples, then show that invalid letters are rejected. Mention that examples are known demonstration sequences, not newly discovered peptides.

**4:00–5:30 — Evidence.** Open Model results. Show the selected model, test metrics, confusion matrix and baseline table. Model selection uses validation AUC. Test data is not used for fitting weights or selecting models. Explain that exact duplicates are removed but similar sequences may cross splits.

**5:30–6:30 — Architecture.** Explain encoder, latent space, decoder, CNN, and random-forest baseline. Open the walkthrough notebook or the small Python modules.

**6:30–7:00 — Limitations and conclusion.** “We have completed the computational pipeline and its interface. We have not performed laboratory experiments. Future work includes similarity-clustered splits, calibration, species-specific activity/MIC models, stronger representation learning and experimental validation.”

## Member responsibilities

| Member | Own and explain | Read first |
|---|---|---|
| Chirag | VAE, preprocessing, integration | `networks.py`: PeptideVAE; `training.py`: train_vae |
| Hurshvardhan Rao | Activity/toxicity models and metrics | PeptideCNN; train_classifier; Model results |
| Aniket Mishra | Generation filters, ranking, visualization | `pipeline.py`; `app.py`; generation audit |

All three should be able to explain the entire flow in one minute.

## Essential concepts in simple language

**Peptide:** a sequence of amino acids represented by 20 standard letters.

**Tokenization:** map each residue to an integer; padding makes batches rectangular. Padding is excluded from sequence reconstruction loss. The VAE has separate START and END tokens.

**Embedding:** a small learned vector for each residue. These are learned from our data, not pretrained ESM-2 embeddings.

**LSTM:** a recurrent network that reads sequence order and carries a memory state.

**VAE:** an encoder predicts a distribution instead of a single code. A decoder learns to reconstruct sequences from samples of that distribution. At generation time, draw z from a normal distribution and decode it.

**Reparameterization:** `z = mu + exp(0.5 * logvar) * epsilon`. Random noise is separated from the learned mean and variance so gradients can train the encoder.

**Loss:** reconstruction cross-entropy measures next-token errors; KL loss encourages the latent distribution to resemble the normal prior. We gradually increase KL weight to 0.02. This is a beta-VAE-style weighted objective, not unweighted maximum-likelihood training.

**Teacher forcing:** during training, the decoder sees true previous residues. During generation, it sees its own previous predictions. Consequently teacher-forced token accuracy is easier than generating a complete correct sequence.

**Word dropout:** hide some decoder inputs during training so it has an incentive to use latent information.

**CNN:** scans short windows of 3, 5 and 7 residues for patterns. Max pooling summarizes the strongest patterns, and a classifier combines them with sequence descriptors.

**Random forest:** many decision trees voting on 25 simple descriptors. It is a baseline, and its importance plot helps discuss which descriptors the data associates with labels.

**Ensemble:** average the CNN and forest scores. The validation set decides whether to deploy CNN, forest, or their average for each task.

**Training/validation/test:** learn parameters / choose checkpoint and model / estimate final performance. We retain a published test split and remove exact overlapping sequences from training.

**ROC-AUC:** how well scores rank positives above negatives across thresholds. **Precision:** fraction of predicted positives that are actually positive. **Recall:** fraction of true positives detected. **F1:** harmonic mean of precision and recall. **MCC:** balanced correlation-like measure using all confusion-matrix cells.

**Priority:** `100 * activity_score * (1 - toxicity_score)`. Both criteria matter; neither score is a calibrated real-world probability.

## Likely questions

**Did you achieve the charter?** The computational generation → activity classification → toxicity classification → ranking pipeline and frontend are implemented. We use compact learned embeddings rather than the proposed ESM-2 representations; this is an explicitly documented architecture change. Activity is binary classification, not MIC regression.

**Why are activity results so high?** The benchmark is easier than real-world discovery: positive and negative sequences can differ substantially in length and composition, and homologous sequences can cross splits. A high AUC is a benchmark result, not a clinical claim.

**Are the generated peptides new?** They are filtered against exact sequences in GRAMPA and classifier training data. That does not prove uniqueness across all databases, dissimilarity to existing sequences, or useful biological function.

**Does low predicted toxicity mean safe?** No. It means lower support for the benchmark's toxic class. Other forms of toxicity and experimental conditions are outside the model.

**What prevents leakage?** Exact deduplication, conflicting-label removal, train/test overlap removal, training-only scaler fitting, and validation-only model selection. Similarity-based leakage remains a limitation.

**What if generation returns fewer sequences?** The app reports how many passed quality filters from the bounded sample pool. Widen the length range or change the seed; it does not invent replacements.

**Why not report just accuracy?** Class imbalance can make accuracy misleading. We also report precision, recall, F1, MCC, ROC-AUC, average precision, and the confusion matrix.

**How is this different from the original notebook?** The original 200-sequence VAE often produced repeated residues. The final implementation trains on a larger cleaned subset, uses stochastic sampling, latent conditioning, dropout, KL warmup, independent predictors, explicit evaluation and an integrated UI.

**Was AI used?** Yes. AI assisted implementation, debugging and documentation. The team should disclose this under the charter and personally understand the code, data, results and limitations. Do not claim manual authorship of every line or claim results from experiments that were not performed.
