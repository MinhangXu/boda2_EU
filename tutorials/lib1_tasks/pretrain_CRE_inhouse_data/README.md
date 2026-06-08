# Pretrained CRE Models On In-House Data

This directory is for notebooks that apply public-data pretrained CRE
checkpoints to in-house Lib1 data before a dedicated from-scratch or fine-tuning
workflow exists for that part class.

Use this folder when the biological question is:

- does a promoted public checkpoint transfer at all to an in-house Lib1 assay?
- is one-shot scoring good enough to justify a fine-tuning pipeline?
- which part class should receive the next scratch/fine-tune comparison?

## Current Canonical Notebooks

| Notebook | Role | Current status |
|---|---|---|
| `promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | One-shot in-house promoter and intron evaluation using promoted public checkpoints. | canonical diagnostic |

## Boundary With Neighboring Folders

- Public-data HPO and checkpoint-promotion notebooks stay in
  `../pretraining_CRE_public_data/`.
- Dedicated fine-tuning notebooks should move under
  `../fine_tuning/<part_class_or_project>/`.
- Reusable scoring or plotting code should move into `src/analysis/` or
  `src/finetune/` once it is called by more than one notebook.

Generated prediction tables, checkpoints, and per-run folders should stay out of
Git unless they are small decision artifacts.
