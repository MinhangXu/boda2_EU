# Multi-Part Combinatorial Training Strategy

Generated: 2026-06-09

This plan starts after the single-part modeling scaffold is clean enough to be
usable as context. It connects enhancer, promoter, 5 Prime UTR, intron, and
3 Prime UTR modeling decisions to future combinatorial GRE training.

## Current Single-Part Status

| Part class | Pretraining | Fine-tune | From scratch | Immediate next question |
|---|---|---|---|---|
| Enhancer | BODA2/Malinois checkpoint available | In-house Lib1 fine-tune mostly complete | Weak current evidence | Run/commit the cleaner scratch comparison only if needed; otherwise use transfer as the baseline. |
| Promoter | Legacy in-house e7/e30 pretraining exists, but needs a split-safe rerun | Pending | June Lib1 scratch HPO complete for ResNet1D and BassetVL | Decide modal50 versus allvalid/51-padded input policy, then compare scratch against legacy-to-Lib1 fine-tune. |
| 5 Prime UTR | Hani Lib1/Lib2 pretraining complete | Phase 2 and June in-house HPO complete | Phase 3 scratch complete | Decide whether Phase 2 fine-tune or Phase 3 scratch is the better seed for combinatorial work. |
| Intron | Seelig pretraining complete | Pipeline being figured out | June Lib1 modal80 ResNet scratch HPO complete | Confirm architecture/RC and decide whether Seelig fine-tune adds value. |
| 3 Prime UTR | Hani pretraining complete | Pending | June Lib1 modal100 ResNet/Basset scratch HPO complete | Synthesize BassetVL focused RC factorial and compare against deliberate length-context/fine-tune branches. |

## Training Scenarios

### Scenario 1: Segmented Combinatorial From Scratch

Input design:

- one input per GRE part class: enhancer, promoter, 5 Prime UTR, intron,
  3 Prime UTR
- independent shallow convolutional stems per part
- learned fusion layers above part encoders

Use when the combinatorial dataset is large enough to learn interactions
without leaning heavily on public single-part pretraining.

### Scenario 2: Pretrained Part Encoders With New Top Layers

Input design:

- initialize each part encoder from its best single-part checkpoint
- freeze or lightly unfreeze encoders
- train a new fusion head on combinatorial outcomes

Use as the most interpretable first combinatorial baseline. This makes it easy
to ask which part class contributes signal and whether the fusion layer learns
cross-part interactions.

### Scenario 3: Full Long-Sequence Training

Input design:

- concatenate the full GRE sequence, likely 1000+ bp
- start by pretraining/fine-tuning on individual part classes where possible
- freeze selected weights, then continue on combinatorial data

Use when positional context and spacing are biologically central. This is the
highest-risk path because length, sample size, and architecture capacity are
all larger.

## Decision Gates

- Do not start scenario 2 until every included part class has a promoted
  checkpoint or a documented reason to use scratch/random initialization.
- Do not start scenario 3 until sequence formatting, padding, masking, and part
  boundary metadata are specified.
- Keep barcode-count policy explicit: record training barcode thresholds and
  held-out barcode thresholds separately for every part class.

## Repo Structure Proposal

- `plan/combinatorial/`: durable strategy and decision records.
- `src/learn/configs/combinatorial/`: from-scratch and long-sequence HPO configs.
- `src/finetune/finetune_sweep_scripts/combinatorial/`: pretrained encoder
  fusion and fine-tune runners.
- `tutorials/lib1_tasks/combinatorial/`: notebooks that compare scenario 1, 2,
  and 3 outputs.

## Near-Term Tasks

- [ ] Finish single-part GitHub inclusion decisions using
  `plan/repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md`.
- [ ] Promote one canonical checkpoint per part class or mark the part class as
  pending.
- [ ] Define the combinatorial input table schema.
- [ ] Decide whether scenario 2 should freeze all part encoders initially or
  allow last-stage unfreezing.
- [ ] Create a smoke config with tiny data and deterministic output paths before
  launching any full HPO.
