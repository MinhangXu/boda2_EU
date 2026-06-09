# Seelig / Rosenberg 2015 Intron Pretraining Plan

This note summarizes how to reuse Rosenberg, Patwardhan, Shendure, and Seelig,
"Learning the Sequence Determinants of Alternative Splicing from Millions of
Random Sequences", for the intron CRE arm of `boda2_EU`.

## Bottom Line

Use the released Seelig data for neural pretraining/HPO in BODA. Treat the
released HAL model as a baseline, feature generator, or teacher model, not as
direct Basset weights.

The released "pretrained model" is the HAL 6-mer additive model. The Kipoi /
Zenodo artifact is `HAL_mer_scores.npz`, a `(4096, 8)` matrix of 6-mer weights
for eight splice-region bins. It is not a PyTorch CNN checkpoint, so it cannot
be loaded with `CNNTransferLearning` as a parent `state_dict` for
`BassetBranched`.

## Plain-Language Dataset Glossary

This dataset is not an expression MPRA in the usual "more RNA means more
regulatory activity" sense. Each random intron sequence is transcribed, spliced,
and then counted by which splice site the reads support. The signal is splice
choice: for a given construct, what fraction of observed spliced reads used one
splice donor instead of another.

- `A5SS`: alternative 5' splice-site library. The construct contains competing
  splice donor outcomes, so the label describes donor-site usage.
- `SD1`: splice donor 1. In the processed A5SS read-count matrix this is column
  `0`.
- `SD2`: splice donor 2. In the processed A5SS read-count matrix this is column
  `44`.
- `p_sd1`: the scalar donor-1 usage target, computed as
  `SD1 reads / total spliced reads` after filtering to designs with at least one
  read.
- `p_sd2_conditional`: a helper scalar computed as `SD2 / (SD1 + SD2)` when
  either SD1 or SD2 has reads. It is materialized for analysis, but the first
  BODA HPO used `p_sd1`.
- `A5SS SD1 scalar task`: a one-output regression task where the model sees the
  101 bp A5SS sequence and predicts `p_sd1`.
- `A5SS donor distribution task`: the richer future task where the model
  predicts the full normalized distribution over 81 donor outcomes rather than
  only `p_sd1`.

So when this note says "activity" or "signal" for the Seelig intron data, read
it as splice-site usage inferred from read fractions, not promoter/enhancer
expression.

## Primary Sources Checked

- Paper: https://doi.org/10.1016/j.cell.2015.09.054
- Original code/data repo: https://github.com/Alex-Rosenberg/cell-2015
- GEO series: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74070
- Kipoi HAL page: https://kipoi.org/models/HAL/
- Zenodo HAL model files: https://zenodo.org/records/1466088

## Released Assets

### HAL Model

Zenodo/Kipoi provide `HAL_mer_scores.npz`:

- file: `HAL_mer_scores.npz`
- license: MIT
- contents: `weights`
- shape: `(4096, 8)`
- k-mer order: original `dnatools` uses `['A', 'T', 'C', 'G']`, not BODA's
  `['A', 'C', 'G', 'T']`
- model idea: score all 6-mers in local windows around a splice donor, sum
  weights by one of eight positional bins, and transform score differences
  into predicted splice usage.

This can be wrapped in BODA, but it should be a separate model class or
preprocessing transform. It should not be presented as a CNN pretrained
checkpoint.

### Processed Data

The original GitHub repo includes processed sequence and read-count files under
`data_gz/`.

Local inspection of `/tmp/cell-2015/data_gz` after a shallow clone:

| File | Rows | Sequence length | Main use |
| --- | ---: | ---: | --- |
| `A5SS_Seqs.csv.gz` | 265,137 | 101 bp | alternative 5' splice site library |
| `A3SS_Seqs.csv.gz` | 2,211,739 | 80 bp | alternative 3' splice site library |
| `Reads.mat.gz` / `Reads.mat` | A5SS `(265137, 304)`, A3SS `(2211739, 606)` | sparse count matrices | per-position spliced-read counts |

Useful nonzero-row counts:

- A5SS: 265,044 sequences with reads, median 44 reads among nonzero rows.
- A3SS: 1,686,096 sequences with reads, median 2 reads among nonzero rows.

The raw FASTQs are available through GEO/SRA (`GSE74070`, `SRP064967`), but
for BODA model development the processed GitHub/GEO count files are the right
starting point.

## Paper Target Constructions Worth Reproducing

The notebooks in the original repo make several distinct modeling datasets.
They should not be collapsed into a single vague "intron activity" target.

### A5SS SD1 Scalar Task

Notebook: `Cell2015_N6_A5SS_Model_Learning_Curves.ipynb`

- input: A5SS sequences
- features in paper: 3-mer to 7-mer counts from degenerate regions
- target: `p(SD1)`, represented as `[1 - p(SD1), p(SD1)]`
- split: 90/10 train/test in the paper learning curve
- purpose in BODA: first smoke test and paper-comparable baseline

This is the easiest BODA entry point: train `BassetBranched` or `BassetVL` as
a single-output scalar regressor on `p(SD1)`.

### A5SS Donor Distribution Task

Notebook: `Cell2015_N7_A5SS_Model.ipynb`

- input: A5SS sequences
- target: normalized read distribution over splice donor outcomes
- notebook target construction: `Y = hstack((A5SS_data[:, :80], A5SS_data[:, -1:]))`
- output dimension: 81
- paper comparison: HAL reports strong cross-validated prediction for original
  donors and new donor-site usage.

This is the richer pretraining task. It probably needs a categorical /
distributional training wrapper:

- model output: logits with shape `[batch, 81]`
- loss: KL-divergence or cross-entropy against normalized read fractions
- metrics: per-output Pearson R2 for `SD1`, `SD2`, `SDCRYPT`, aggregate new
  donor positions, and distribution KL

`CNNBasicTraining` can handle multi-output regression, but for this target it
would be cleaner to add a `CNNDistributionTraining` graph rather than train
soft distributions with plain MSE.

### A3SS / A5SS Joint Exon Definition Task

Notebook: `Cell2015_N9_Training_Joint_A5SS_A3SS_Model.ipynb`

- A3SS target: splice acceptor 1 usage, `A3SS[:, 235] / total_reads`
- A5SS target: `SD2 / (SD1 + SD2)`
- shared feature idea: 6-mer counts from exonic randomized regions
- purpose: learns shared enhancer/silencer motif effects across A3SS and A5SS

This is useful biologically, but less directly aligned with intron-region
pretraining than the A5SS donor-distribution task.

## Recommended BODA Integration

### Stage 0: Data Materialization

Add a downloader/preparer, for example:

- `src/learn/prepare_seelig_splicing_dataset.py`

The script should:

1. download or read `A5SS_Seqs.csv.gz`, `A3SS_Seqs.csv.gz`, and `Reads.mat.gz`
2. materialize tidy parquet/CSV files under `opt_EU_learn_n_design/introns/seelig_2015/`
3. include explicit columns:
   - `library`: `A5SS` or `A3SS`
   - `seq`
   - `read_count_total`
   - `fold`: train/val/test or partition ID
   - scalar targets such as `p_sd1`, `p_sa1`, `p_sd2_conditional`
   - optional vector targets stored separately as `.npz` or long-form tables

### Stage 1: BODA Smoke Baseline

Add a DataModule for the A5SS SD1 scalar task:

- `boda/data/seelig_splicing_datamodule.py`
- class: `SeeligA5SSScalarDataModule`
- model: `BassetBranched` with `n_outputs=1`, or `BassetVL`
- input length: 101 bp
- graph: `CNNBasicTraining`
- loss: `MSELoss`
- checkpoint monitor: `epoch_end_val_pearson_r2`
- evaluation metrics: MSE, Pearson R, Spearman rho, Pearson R squared, and
  coefficient-of-determination R2

This gives a fast answer to: "Can a Basset-like neural model match the simple
HAL/k-mer baseline on the paper's easiest target?"

### Stage 2: Paper-Comparable Rich Pretraining

Add the A5SS 81-output distribution task:

- class: `SeeligA5SSDonorDistributionDataModule`
- model: `BassetBranched`, `n_outputs=81`
- graph: ideally `CNNDistributionTraining`
- loss: `KLDivLoss` over normalized read-fraction targets
- checkpoint metric: mean of selected paper-comparable Pearson R2 values

This is the best candidate pretraining source for downstream in-house intron
fine-tuning, because it teaches local splice-site usage rather than just a
single scalar.

### Stage 3: Transfer to In-House Introns

Once a BODA neural model is pretrained on the Seelig A5SS task, export it using
the normal BODA artifact path:

- `local_artifacts/introns/seelig_2015/.../model_artifacts__*.tar.gz`

Then downstream in-house intron fine-tuning can use the existing
`CNNTransferLearning` path:

- `parent_weights=/path/to/model_artifacts__*.tar.gz`
- initialize a new intron model with the same compatible trunk dimensions
- replace or resize output heads to match in-house intron targets
- freeze transferred layers for a small number of epochs, then unfreeze

## What To Do With HAL Directly

HAL is still valuable:

1. baseline: report HAL on the same train/val/test splits as BODA neural models
2. teacher: pretrain a Basset model to mimic HAL scores before supervised
   training on Seelig labels
3. feature: append HAL scalar/vector scores to downstream analysis tables
4. diagnostic: compare Basset saliency/motifs against the HAL 6-mer weights

But HAL should not be the parent model for `CNNTransferLearning`, because the
current BODA transfer path expects neural weights with matching module keys.

## Initial HPO Recommendation

Start narrow:

- task: A5SS SD1 scalar
- input length: 101
- model: `BassetBranched` or `BassetVL`
- outputs: 1
- no reverse-complement augmentation, because splice signals are directional
- compare:
  - HAL/k-mer paper baseline
  - Basset scratch HPO
  - optional HAL-teacher initialized Basset

Then move to the richer 81-output task only after the scalar task is
reproducible.

## Implementation Checklist

- [ ] Add `prepare_seelig_splicing_dataset.py`
- [ ] Add `SeeligA5SSScalarDataModule`
- [ ] Register new DataModule in `boda/data/__init__.py`
- [ ] Add `introns/seelig_2015/basset_branched/` sweep config
- [ ] Run a small scalar smoke train
- [ ] Add HAL baseline scorer around `HAL_mer_scores.npz`
- [ ] Decide whether to add `CNNDistributionTraining` for the 81-output task
- [ ] Pretrain and archive the best Seelig neural artifact
- [ ] Fine-tune that artifact on in-house intron data
