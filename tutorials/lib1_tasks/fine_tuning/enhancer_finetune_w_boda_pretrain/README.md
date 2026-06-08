# Enhancer Fine-Tuning Analysis Index

This folder keeps the current canonical enhancer Lib1 fine-tuning notebooks.
Older exploratory notebooks live in the local-only `archive/` folder, which is
ignored by Git.

## Canonical Notebooks

| Notebook | Question | Status |
|---|---|---|
| `barcode_range_comparable_bins_decision_analysis_may2026.ipynb` | Equal-N barcode-bin training comparison. | canonical |
| `may13_2026_bc_training_eval.ipynb` | Barcode count in the training set. | canonical |
| `may13_2026_bc_test_eval.ipynb` | Barcode count / measurement quality in the held-out test set. | canonical |
| `may15_2026_hq8_multiseed_hpo_analysis.ipynb` | HQ8 multiseed HPO and current enhancer decision surface. | canonical |

## Archive Policy

The local `archive/` folder contains superseded notebooks for early transfer,
HQ-first, random-split, scratch, and first-pass barcode analyses. Those notebooks
can be inspected locally, but they are not part of the GitHub update unless a
specific historical audit requires them.

Generated training roots remain under ignored `src/finetune/learning_curve/`.
Delete them only after the relevant takeaway is captured in
`plan/repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md`
or the enhancer rehydration plan.
