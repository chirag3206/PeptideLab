> Final update: also read [FINAL_ALIGNMENT_AND_TEAM_GUIDE.md](FINAL_ALIGNMENT_AND_TEAM_GUIDE.md). Frozen ESM-2 + CNN, validation-selected property flags, selectable weighted ranking and similarity/composition audits are now implemented. The compact baseline remains the validation-selected default.

# Data sources and label meanings

## GRAMPA

Source: https://github.com/zswitten/Antimicrobial-Peptides

The supplied raw CSV has 51,345 records and 6,760 unique canonical sequences. Records include sequence, organism, strain, modification information, an activity value, unit and source identifiers. The final generator consumes sequences only. It does not use species, modifications or measured activity as conditioning variables. Dataset membership is not converted to a toxicity label.

`python -m peptide_lab.preprocessing` recreates the canonical unique pool and both length subsets without overwriting historical files. The report records raw and processed hashes. The full generator uses 5–60 residues; the new toy study uses 5–40.

## Activity benchmark

Immediate source: UniDL4BioPep repository, `5. Antimicrobial activity/AMP_train.xlsx` and `AMP_test.xlsx`, pinned at the commit in `Data/external/sources.json`.

Target: binary antimicrobial-class membership; 1 is positive, 0 is negative. This is not a minimum inhibitory concentration prediction or a species-specific susceptibility estimate. Actual local labels determine class counts. Invalid/noncanonical sequences and lengths outside 5–200 are excluded before exact deduplication and splitting. Detailed counts are in `Results/final/activity_data_audit.json`.

The deployed model averages a learned-residue CNN and a descriptor-based random forest, selected by validation AUC. It does not use pretrained ESM-2. Positive and negative sequence lengths differ strongly. A length-only classifier and short-subset class counts quantify this bias in the extended evaluation; the short test subset has only eight negatives. Predictions on training/validation rows are demonstrations, not held-out performance evidence.

## Toxicity benchmark

Immediate source: the same pinned repository, `17. Toxicity 2021 Dataset/train_3284.xlsx` and `test_580.xlsx`.

Target: the benchmark's binary toxic/non-toxic classification; 1 is toxic, 0 is non-toxic. The spreadsheets provide sequence and label, not dose, target tissue, assay conditions or organism-specific toxicity endpoints. Consequently the app must not interpret this score as hemolysis, clinical safety or an all-purpose toxicity assessment.

The UniDL4BioPep paper identifies ATSE as its toxicity benchmark reference. Its table lists balanced splits, whereas the downloaded pinned files contain 1,663 positives/1,621 negatives in training and 269 positives/311 negatives in test. We preserve and report the actual file labels; we do not claim an exact replication of the paper's evaluation or substitute the paper's class counts. Original assay-level provenance is not recoverable from the two-column spreadsheets alone.

Reference: Du et al., *UniDL4BioPep: a universal deep learning architecture for binary classification in peptide bioactivity*, DOI 10.1093/bib/bbad135. [Author-hosted paper](https://www.grains.k-state.edu/research/ccl/files/publication_pdf/2023_BIB_Du_UniDL4BioPep.pdf), Table 1 and reference 46 (ATSE, DOI 10.1093/bib/bbab041).

## Shared evaluation policy

Training, validation and test sets have no identical sequences in common within each task. Similar sequences may still cross splits. No similarity-disjoint or independent external cohort evaluation is claimed. Classification uses threshold 0.5. Scores are uncalibrated model outputs. Brier/reliability diagnostics assess error but do not turn the deployed outputs into validated biological probabilities.

The dataset explorer displays each model's split membership separately and joins labels by sequence. A missing label stays missing. GRAMPA sequences may overlap classifier datasets; there is no claim that browsing a known peptide constitutes novel discovery.
