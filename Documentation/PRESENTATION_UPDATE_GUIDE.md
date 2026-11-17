# Presentation update guide

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
