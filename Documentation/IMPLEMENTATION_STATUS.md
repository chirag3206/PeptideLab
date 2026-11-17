> Final update: also read [FINAL_ALIGNMENT_AND_TEAM_GUIDE.md](FINAL_ALIGNMENT_AND_TEAM_GUIDE.md). Frozen ESM-2 + CNN, validation-selected property flags, selectable weighted ranking and similarity/composition audits are now implemented. The compact baseline remains the validation-selected default.

# Current implementation status

Updated 17 November 2026. This supersedes implementation-status statements in the earlier read-only handover audit; historical audit observations remain preserved.

| Component | Verified implementation |
|---|---|
| Raw GRAMPA preprocessing | Implemented and executed; reconstructs the canonical pool and 5–40/5–60 subsets; exact original sequence order retained |
| Full VAE | Existing deployed weights preserved; training and reconstruction/generation evidence available |
| Latent diagnostics | Executed zero/shuffled-vector controls and free-running reconstructions on validation data |
| Controlled toy study | 22 completed runs; ten-seed robustness, matched latent dimensions and word-dropout comparison |
| Activity prediction | CNN/forest ensemble with saved independent task test metrics |
| Toxicity prediction | CNN with saved independent task test metrics |
| Predictor diagnostics | Length-only baseline, short-subset counts/balanced accuracy, bootstrap intervals, errors and reliability bins |
| Ranking | Product rule implemented; alternative preference sensitivity measured on saved candidate pool |
| Dataset explorer | GRAMPA/activity/toxicity sources, filters, single/multiple selection, seeded sampling, known-label comparisons |
| Custom input | Sequence/FASTA text and CSV/FASTA upload; input validation |
| Generated candidates | VAE sampling, quality/novelty filters, both predictors and ranking |
| Prediction inspection/history | Individual scores/descriptors, available labels, per-task split membership, latest 10 session runs |
| Exports | Candidate CSVs, filtered dataset CSVs, self-contained HTML report, executed notebooks |
| Provenance | Source checksums, pinned downloads, source/data/model evidence manifest; retraining archives |

## What is not established

- Missing original full-VAE/02B notebooks and original research-context files have not been recovered. New studies do not masquerade as historical results.
- Exact paper reproduction and biological superiority over the historical model are not claimed.
- Similarity-disjoint external validation is not performed; exact duplicate separation alone is weaker.
- Scores are not calibrated biological probabilities. Reliability metrics diagnose error but do not establish clinical calibration.
- Toxicity labels do not specify dose, assay or tissue in the supplied spreadsheets. Activity is binary membership, not MIC or species-specific susceptibility.
- No wet-lab testing is performed. Novelty is an exact-match property within stated sequence pools.
- Three-seed hyperparameter comparisons are exploratory; they do not establish optimal full-data settings.

The application is a complete runnable computational demonstration with documented evaluations. The limits above are scientific claims it deliberately does not make.


Updated generation: choose a raw sample pool size. Every passed-filter peptide is scored, ranked, plotted and exported. Rank 1 is best. No accepted sequences are truncated.
