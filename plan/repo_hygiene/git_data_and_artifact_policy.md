# Git Data And Artifact Policy

**Repository:** `MinhangXu/boda2_EU`

**Current GitHub remote visibility:** public fork

**Development target:** current public repository with code-only, data-excluded checkpoints

**Policy status:** living repository policy
**Adopted:** 2026-07-21

## Purpose

Git is the public history of the code and scientific decisions. It is not the
storage system for unpublished biological data, exact held-out memberships,
large model products, W&B caches, or generated analysis trees.

The default public commit contains:

- source code, tests, launch logic, and configuration;
- protocols, decision records, and human-readable conclusions;
- source notebooks only after outputs and execution state are cleared;
- schemas, algorithms, seeds, thresholds, aggregate counts, and checksums that
  do not reveal row-level private data; and
- deliberately reviewed, compact public metadata.

Everything else stays outside public Git unless it passes an explicit review.

The July 2026 work may be committed and pushed on the current branch after the
complete outgoing commit range is reviewed for unpublished data, exact split
memberships, generated artifacts, secrets, and machine-specific paths. Private
repository migration is optional and deferred; the current development
boundary is enforced by artifact classification, narrow staging, ignore rules,
and pre-push review.

## Classification And Default Location

| Artifact class | Public Git default | Storage and rationale |
|---|---|---|
| Python, shell, YAML, tests, and Markdown | Track | These define behavior and scientific intent. |
| Raw or derived biological tables (`.csv`, `.tsv`, count matrices, sequence tables) | Ignore | Keep in access-controlled lab storage or regenerate from approved inputs. |
| Exact Lib1 split manifests | Ignore | They expose exact sequences, barcode support, development folds, and final-test membership. |
| Sanitized manifest summary | Review, then track | May contain schema version, algorithm, seed, thresholds, aggregate counts, and approved SHA-256 values, but no row IDs, sequences, memberships, or absolute private paths. |
| Full run registries and per-epoch histories | Local/private by default | They are generated operational state and may contain local paths or unpublished run provenance. |
| Curated `best_runs.csv` | Track as an explicit exception | It is a small, manually reviewed model-selection index used by repository code. |
| Checkpoints, model archives, embeddings, predictions, and W&B state | Ignore | Store in W&B or access-controlled artifact storage and identify them by immutable ID/hash. |
| Generated plots, rendered notebooks, HTML, and analysis output trees | Ignore by path | Keep source programs/notebooks in Git. Promote only a deliberately selected, public-safe figure through review. |
| Identifiable meeting transcripts | Ignore | Keep in approved private meeting storage; commit a reviewed decision/action summary instead. |
| Draft compiled reports | Ignore | Track the report source and approved illustration assets; archive a deliberately frozen final report separately if needed. |
| Large public artifact that is not sensitive | External by default | Size alone can use an artifact store; Git LFS does not make sensitive data private. |

## What The Exact Split Manifest Does

The split manifest is a frozen experiment **seating chart plus receipt**. It:

1. binds a run to an exact dataset SHA-256 and preprocessing contract;
2. assigns every stable construct ID to `train_only`, one development fold, or
   the locked `audit_test` partition;
3. prevents train/development/final-test leakage across campaign stages;
4. makes splits independent of input row order; and
5. lets verifiers detect changed, incomplete, or misjoined data before model
   training or evaluation.

The model does not read its input sequence from the manifest. It reads the
sequence from the learn-ready local dataset. The manifest stores a second
exact sequence and barcode count so the DataModule can compare them with the
dataset and fail if a stable ID points to the wrong sequence or support value.

The stable construct ID is a SHA-256 hash of `parts_concatenated`. That is a
deterministic join key, not anonymization: someone who knows a candidate
construct can hash it and test whether it occurs in a published partition.
Therefore neither the exact sequences nor the hashed row-level memberships
belong in this public repository.

The private generated manifest root is:

```text
src/learn/data_manifests/
```

The public repository keeps the generator, verifier, schema logic, tests, and
scientific protocol. An authorized environment restores or regenerates the
private manifest bundle and verifies its SHA-256 before use.

## Ignore Rules And Exceptions

`.gitignore` now treats tabular datasets and machine-generated result tables as
local by default, including `.csv`, `.tsv`, `.jsonl`, spreadsheet, array, and
database formats. It also excludes the exact manifest root and established
generated output/plot roots.

There are narrow exceptions for source-like synthetic fixtures and the curated
`src/learn/run_registry/best_runs.csv`. A new ignored file may enter Git only
after review confirms all of the following:

- it is necessary for a clean clone to understand or test the repository;
- it contains no unpublished sequence, count, row-level split, or private path;
- a compact schema/example/Markdown summary cannot serve the same purpose;
- its size is appropriate for ordinary Git; and
- the commit message explains why the exception is public and durable.

After review, use a narrow negation rule or an explicit `git add -f <path>`.
Never use `git add -A` or `git add .` in this repository's dirty research
worktree.

## Existing Tracked Tables

Ignore rules apply to untracked files. They do not silently remove files that
Git already tracks. This is intentional for the July 2026 checkpoint: existing
fixtures, historical learning-curve tables, and run-registry CSVs remain
visible until they receive a separate migration review.

To stop tracking an existing file while keeping the local copy requires an
explicit operation such as:

```bash
git rm --cached -- path/to/file.csv
```

That deletion must be reviewed and committed. It does not erase the file from
older public commits. Removing sensitive content that was already published
requires a coordinated history rewrite and credential/data incident review;
`.gitignore` is not a history-purge mechanism.

The full `runs.csv` and `sweep_launches.csv` are active inputs to several
current tools. Do not untrack them casually. A later migration should first
provide a private/local full registry plus a compact redacted public snapshot
or another tested replacement. `best_runs.csv` remains the intentional public
registry.

## Storage, Restoration, And Verification

Every private artifact needed to reproduce a result should have:

- a durable access-controlled storage location owned by the lab/project;
- a logical artifact name and schema/version;
- a SHA-256 checksum;
- the producing code commit and command;
- the source-data version or hashes; and
- a short restoration note that does not expose credentials.

Do not put access tokens or signed download URLs in Git. On a new machine,
restore the private bundle separately, place it at the documented local path
or configure an environment variable, and verify the checksum before running
the corresponding campaign verifier.

## Branches And Merges

A branch is a movable label pointing to a chain of Git commits. It is not a
second physical copy of the research data. Creating or switching branches does
not remove ignored local files; Git simply does not version or merge them.

Branches merge:

- code;
- tests;
- configuration;
- protocols and status documentation; and
- public-safe summaries and hashes.

Branches do **not** merge ignored datasets, exact manifests, checkpoints, or
generated result trees. Those artifacts stay in approved external storage and
are restored or regenerated independently after the code merge.

For the current July 2026 checkpoint, continuing on
`checkpoint/learn-finetune-docs-may2026` is the simplest option because the
uncommitted July work directly continues that branch and there is no need to
merge one checkpoint branch into another. Make small scoped commits there,
review them locally, and push only after the full outgoing range passes the
public-data safety review. A new branch is useful only when the work needs an
independent review boundary, a risky experiment, or parallel development.

## Optional Future Private-Development Migration

If the project later adopts a private development repository:

1. choose a standalone private repository, preferably owned by the project or
   laboratory organization so team roles can be managed centrally;
2. confirm the institution's research-data and approved-AI-service rules;
3. grant only the required people and GitHub/Codex application access to that
   repository;
4. configure and test authenticated fetch and push from this workstation;
5. retain the existing public fork only as an upstream-code reference, unless
   a separately reviewed detachment is preferred; and
6. review the complete outgoing commit range for private data,
   generated artifacts, and machine-specific paths.

This remains an optional future administrative checkpoint. It is not required
for reviewed code-only checkpoints that comply with this public Git policy.

## Pre-Commit Checklist

Before every checkpoint commit:

1. inspect `git status --short` and stage named paths only;
2. inspect `git diff --cached --stat` and `git diff --cached`;
3. confirm no exact sequence, row-level split membership, raw counts, private
   path, token, checkpoint, or oversized generated table is staged;
4. confirm source notebooks have no embedded execution output;
5. run the relevant unit/static verification suite;
6. record public conclusions in Markdown rather than copying full result
   tables; and
7. push only after the staged tree and commit series have been reviewed.
