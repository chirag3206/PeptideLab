> Final update: also read [FINAL_ALIGNMENT_AND_TEAM_GUIDE.md](FINAL_ALIGNMENT_AND_TEAM_GUIDE.md). Frozen ESM-2 + CNN, validation-selected property flags, selectable weighted ranking and similarity/composition audits are now implemented. The compact baseline remains the validation-selected default.

# Project decisions and evidence

- **[Supported Decision] Keep deployed weights stable.** The dashboard extension and research diagnostics do not replace the models used for the working demonstration. Tests verify hashes against the earlier manifest.
- **[Supported Decision] Separate known labels from predictions.** Dataset browsing joins by exact sequence and preserves missing values. Activity and toxicity splits are displayed independently; a training-row match is not a test result.
- **[Supported Decision] Keep both length ranges explicit.** Full VAE: 5–60, 6,510 sequences. New controlled toy study: 5–40, 6,154-sequence pool. No claim that the historical and current VAE runs are matched.
- **[Direct Observation] Latent-vector controls change decoder loss.** Validation cross-entropy is approximately 1.729 with encoder mean, 3.594 with zeros and 4.420 with shuffled means. This supports latent dependence under the fixed protocol, not biological semantics.
- **[Direct Observation] Activity has a strong length shortcut.** A training-only length baseline attains approximately 0.967 test AUC. Only eight negative examples occur in the 5–40 activity test subset; display counts and balanced accuracy alongside accuracy.
- **[Supported Decision] Do not tune the deployed model from these post-hoc tests.** Extended results explain limitations. They are not a new untouched benchmark.
- **[Supported Decision] Explore one factor at a time in the toy study.** Match sample and split within seed while varying latent size or word dropout. Separate the 10-seed robustness mean from the matched 3-seed baseline. Bootstrap intervals with 3 runs are weak evidence.
- **[Supported Decision] Preserve the product ranking rule.** Alternative weight settings are sensitivity examples, not inferred historical weights or optimized clinical utility.
- **[Supported Decision] Use CPU inference for the demonstration.** Existing installed dependencies and trained weights work without internet or GPU configuration.
- **[Supported Decision] Preserve reproducibility artifacts.** Pin download revisions, hash source/data/model inputs, retain per-run outputs and archive existing production artifacts before deliberate retraining.


Updated generation: choose a raw sample pool size. Every passed-filter peptide is scored, ranked, plotted and exported. Rank 1 is best. No accepted sequences are truncated.
