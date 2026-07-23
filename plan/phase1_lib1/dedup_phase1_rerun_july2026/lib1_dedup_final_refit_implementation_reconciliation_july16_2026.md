# Lib1 Dedup Final-Refit Implementation Reconciliation

**Date:** 2026-07-16
**Status:** frozen before the checkpoint allowlist and before any authorized
audit loader was instantiated
**Scope:** administrative and isolation-implementation reconciliation only;
the five selected policies, three seeds, epoch budgets, primary predictor,
metrics, Intron views, and prohibition on audit-driven reselection remain
unchanged.

## Why this note exists

The July 16 final-refit/audit amendment was intentionally strict. A post-run,
pre-audit code reconciliation found two literal implementation differences and
one provenance limitation. None exposed an audit target, sequence, prediction,
metric, or stratum result, and none can alter a scientific decision. They are
recorded here rather than silently described as exact compliance.

## 1. Stable-ID-only exclusion metadata

The amendment called the refit exclusion a positive non-audit allowlist and
said the refit must not enumerate complementary audit rows. The implemented
`final_refit` mode did not construct an audit dataset or loader and did not
parse audit sequence, target, or barcode fields from the source table. It did,
however:

1. read the frozen split JSON assignments;
2. read the source table's stable-ID column only;
3. resolve the audit IDs to CSV row offsets; and
4. skip those offsets before pandas parsed the full training columns.

Thus the stronger scientific isolation condition held—no audit sequence,
target, barcode value, prediction, or performance result entered refitting—but
the literal "do not enumerate/count" wording did not. The allowlist must label
this method `stable_id_only_physical_row_exclusion`, retain the frozen audit-ID
hash, and retain the non-audit training/normalization hashes. This exception
does not authorize any additional audit metadata inspection.

## 2. W&B project label

The amendment proposed `<part>__bashor_in_house__dedup_exact_v1__audit_refit`.
The executed, manifest-bound label was instead:

```text
<part>__bashor_in_house__dedup_exact_v1__final_refit_development
```

This label more accurately describes the non-audit fitting process. It changes
only W&B organization, not any command hyperparameter, dataset row, seed,
checkpoint, or selection rule. The exact executed project must be retained in
the allowlist and verified against the SHA-256-bound 15-row refit manifest.

## 3. Completed-epoch evidence

The append-only registry does not have a dedicated measured
`completed_epochs` field for this campaign. Each training process did,
however, fail closed unless `trainer.current_epoch` exactly equaled the frozen
part-specific epoch budget before the final checkpoint and portable artifact
were saved. Every allowlisted row must therefore retain:

- the frozen expected epoch count;
- the reconciled successful completion record;
- the completion marker and training-log path plus SHA-256;
- SHA-256 values for the exact training entry point, DataModule, runner,
  manifest generator, model, and graph source files used by the assertion and
  save path.

This is indirect but deterministic completion evidence. Future campaigns
should add a first-class observed completed-epoch field to artifact provenance.

## Pre-audit consequences

Before audit access, the allowlist/scorer must also:

- bind every row semantically to final-refit manifest SHA-256
  `83ec532cf84e83d3477f2e6e8c716a04284fcc43b7d7c4426338a8b0f093582c`;
- verify exact part/config/architecture/model/graph/RC/loss/seed/epoch/status,
  dataset, split, selection, and protocol identities;
- retain the five selected fold run IDs and, for weighted policies, the
  unweighted mate cell/run provenance when available;
- verify artifact, checkpoint, compact-provenance, completion-marker, log, and
  implementation-source hashes;
- permit only explicitly documented, hash-identical technical retries after
  an incomplete attempt and never after a completed audit.

This reconciliation does not weaken the audit firewall: the audit remains one
confirmatory report over the already-frozen policies, and the results cannot
return to model selection, calibration fitting, or Stage 4 design.
