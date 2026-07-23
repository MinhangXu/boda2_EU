# Evo2 Foundation-Encoded Sequence-to-Expression Regression Plan

## One-line goal

Build a clean benchmark pipeline that uses Evo2 as a **frozen autoregressive DNA language foundation model** to convert each **synthetic DNA construct** into a construct-level embedding, then trains lightweight supervised heads to predict expression outputs such as RNA, protein, or both.

Recommended project name:

**Evo2 Foundation-Encoded Sequence-to-Expression Regression**

Recommended short task name:

**Evo2-Seq2Expr**

---

## Conceptual framing

This is **not** Evo2 fine-tuning in the first implementation. Evo2 is used as a fixed feature extractor / frozen encoder.

The best DL terms for the method are:

- **Frozen-encoder transfer learning**
- **Foundation-model embedding regression**
- **Intermediate-layer probing**
- **Foundation-encoded sequence-to-expression regression**

Biological task framing:

> Given a synthetic DNA construct containing tunable CRE components, learn a supervised mapping from frozen Evo2 sequence embeddings to measured expression phenotypes.

Input:

> **Synthetic DNA construct**, approximately **7 kb** for the intended full Expression Unit use case, with the combined variable regions usually **<2 kb**.

Output targets:

- RNA expression
- Protein expression
- Multi-output RNA + protein expression
- Eventually replicate-aware / uncertainty-aware targets if barcode-level or replicate-level measurements are available

---

## High-level pipeline

**Current concept illustration:**
[`evo2_embedding_prediction_plan.png`](../tac_campaign_july2026/figures/evo2_embedding_prediction_plan.png)

The illustration is conceptual. Before it becomes a finalized methods figure,
update the 1,500-bp construct label to the intended approximately 7-kb
Expression Unit, add the token-position reduction/pooling step, avoid assuming
a four-token tokenizer description, and replace the provisional log10 target
axes with the final RNA/protein target definitions.

```text
Synthetic DNA construct
        |
        v
Frozen Evo2 DNA language foundation model
        |
        v
Candidate intermediate layer hidden state
        |
        v
Construct-level sequence embedding
        |
        v
Trainable supervised head
        |
        v
RNA expression, protein expression, or joint RNA/protein output
```

Important modeling statement:

> Candidate Evo2 layers are evaluated separately. We are not combining all intermediate layers by default. The first search is an **OR** relationship: layer 8 **or** layer 16 **or** layer 28 **or** final layer, etc. The goal is to find which hidden representation works best for this supervised biological task.

---

## Why intermediate-layer probing is necessary

Evo2 is pretrained autoregressively to model DNA sequence. Its final layer may be optimized for next-token/base prediction and genomic likelihood rather than regulatory activity.

Expression prediction may depend on features distributed across depth:

- earlier/middle layers may capture local sequence grammar, motifs, GC content, and k-mer-like patterns;
- middle/late layers may capture longer-range motif combinations or regulatory grammar;
- final layers may be too specialized for Evo2's pretraining objective.

Therefore, use **intermediate-layer probing**:

> Extract embeddings from multiple candidate Evo2 layers, train the same downstream regressor on each candidate representation, and choose the best layer using validation-set performance.

Do not select a layer based on test performance.

---

## Current repository context and constraints

The existing BODA-style training framework expects data modules, model modules, and graph modules to be registered and passed through the trainer CLI. The trainer dynamically retrieves `data_module`, `model_module`, and `graph_module`, instantiates them, runs Lightning training, and saves model artifacts. This means the embedding pipeline should be implemented as a sibling model family that plugs into the same pattern rather than rewriting the whole trainer.

Existing sequence models include Basset/BassetVL-style one-hot CNN models with configurable input length, convolutional layers, linear layers, output count, and loss criterion. These are the natural sequence-based baselines for comparison.

Existing `CNNBasicTraining` assumes batches are shaped as `(x, y)` and directly applies `model.criterion(y_hat, y)`. If we return `(embedding, target, weight)`, we need a separate graph/training module or a safe branch that handles weighted batches explicitly.

## Repository placement decision, updated during implementation

**Decision for the first implementation: keep this inside `boda2_EU`, not a new repo yet.**

Rationale:

- The work needs identical train/validation/test rows, identical target normalization rules, and direct comparison against existing BODA2 BassetVL/ResNet baselines.
- The current trainer already supports pluggable `data_module`, `model_module`, and `graph_module` classes, so frozen Evo2 embeddings can enter as a sibling model family instead of a forked training stack.
- The foundation-model extraction step is the only Evo2-specific piece. That piece should remain compartmentalized under `src/foundation/evo2/`, while cached embeddings are consumed by normal BODA modules under `boda/data`, `boda/model`, and `boda/graph`.

Create a separate repo only after the benchmark stabilizes and one of these becomes true:

- we need to distribute the Evo2 embedding benchmark independently of BODA2;
- Evo2 extraction/training dependencies become too heavy or incompatible with the BODA2 environment;
- the project grows into a general foundation-model sequence regression toolkit rather than a BODA2 baseline/comparison track.

Current compartment boundaries:

```text
src/foundation/evo2/                  # Evo2-specific extraction and artifact creation
src/learn/evo2/                       # BODA-aware probing runners for cached embeddings
boda/data/embedding_datamodule.py     # Generic cached-embedding supervised data module
boda/model/embedding_heads.py         # Generic trainable heads
boda/graph/embedding_prediction.py    # Generic embedding regression Lightning graphs
```

The earlier draft paths that used `src/boda/...` should be read as `boda/...` in this repository.

## Implementation status as of this pass

Implemented scaffold:

- `src/foundation/evo2/extract_evo2_embeddings.py`
  - validates CSV/TSV/parquet input tables;
  - writes `rows.parquet` when a parquet engine is available, otherwise falls back to `rows.csv`;
  - writes one `.pt` embedding artifact per requested layer plus `manifest.json`;
  - has a strict `--backend evo2` path with a clear import error if Evo2 is unavailable;
  - has a `--backend hash_smoke` path for deterministic plumbing tests only, not for science results.
- `boda/data/embedding_datamodule.py`
  - loads cached embeddings and row metadata;
  - aligns by `construct_id`;
  - enforces no duplicate construct ids;
  - supports train-only feature and target standardization;
  - supports optional barcode-derived sample weights.
- `boda/model/embedding_heads.py`
  - adds `EmbeddingMLPRegressor`;
  - adds `EmbeddingHeteroscedasticRegressor`.
- `boda/graph/embedding_prediction.py`
  - adds unweighted MSE training;
  - adds weighted MSE training;
  - adds heteroscedastic Gaussian NLL training;
  - logs normalized keys such as `val_pearson_mean`, `val_cod_r2_mean`, and `val_spearman_mean`.
- `src/learn/evo2/run_evo2_layer_probe.py`
  - runs a fixed MLP probe across cached layer embedding files;
  - writes an incremental `layer_probe_summary.csv`;
  - does not include test metrics unless `--include_test_metrics` is explicitly set.

Registered with the existing BODA lookup:

- `EmbeddingRegressionDataModule` in `boda.data`;
- `EmbeddingMLPRegressor` and `EmbeddingHeteroscedasticRegressor` in `boda.model`;
- `EmbeddingRegressionTraining`, `WeightedEmbeddingRegressionTraining`, and `HeteroscedasticEmbeddingRegressionTraining` in `boda.graph`.

Verification in the current local shell:

- Syntax check passed with `python -m py_compile` for all new Python files.
- A new `boda_evo2_env` conda environment was created by cloning `boda_env` and adding `pytest==7.4.4` plus `einops==0.6.1`.
- Runtime smoke tests pass in `boda_evo2_env`: `python -m pytest tests/test_embedding_pipeline_smoke.py -q`.
- A two-layer `hash_smoke` extraction plus fixed-head layer-probe run completed and wrote a summary CSV under `/tmp/boda_evo2_env_probe/`.
- Hardware checks show the server is good for BODA and cached-embedding heads, but not for current local Evo2 extraction: the driver reports CUDA 12.0, Torch is cu117, GPUs are compute capability 6.0/6.1, and bf16 support is false. Current upstream Evo2 local inference expects a newer Python/CUDA/Torch stack; see `src/foundation/evo2/ENVIRONMENT.md`.

## Hardware refresh plan for local Evo2 extraction

The hardware/software constraint is real and comes from the current upstream Evo2 documentation:

- Evo2 local inference expects Linux, CUDA 12.1+ with compatible drivers, cuDNN 9.3+, GCC 9+/Clang 10+, Python 3.11 or 3.12, and Torch 2.6.x/2.7.x.
- Evo2 20B, 40B, and 1B-base require FP8/Transformer Engine and a NVIDIA Hopper GPU for numerical accuracy.
- Evo2 7B models can run without Transformer Engine in bfloat16, but still need the newer CUDA/Python/Torch stack and an actually supported modern GPU.
- NVIDIA's Evo2 NIM support matrix for the 40B model lists 2x H100 80 GB or 1x H200 144 GB, with driver 535+ and 100 GB disk for container/model storage.
- FlashAttention 2, used by the Evo2 light install path, supports Ampere, Ada, or Hopper GPUs on CUDA. The current Pascal GPUs are older than that support target.

Current local machine:

```text
Ubuntu 20.04.6
Driver 525.125.06 / nvidia-smi CUDA 12.0
Torch CUDA runtime 11.7 in boda_env
GPUs: 1x Tesla P40 24 GB, 1x Tesla P100 16 GB, 6x GTX 1080 Ti 11 GB
Compute capability: 6.0 / 6.1
GPU topology: PCIe only, no NVLink
Host RAM: 251 GiB
/home free space at check time: about 190 GB
```

This cannot be fixed by a conda environment alone. Local real Evo2 extraction requires a GPU refresh plus a newer driver/toolchain.

Important clarification: this limitation applies to **inference / embedding
extraction**, not only fine-tuning. Extracting intermediate hidden states still
loads the checkpoint and runs a forward pass through all layers up to the
requested layer. The official examples use CUDA tensors and the documented local
inference stack depends on CUDA/cuDNN plus model-specific precision support.
There is no documented supported CPU-only Evo2 embedding extraction path. A
CPU-only experiment might be theoretically possible with enough engineering, but
it would be very slow, would not match the released GPU inference path, and is
not the route to use for this benchmark.

### Transformer Engine / 7B runtime decision

We do **not** need to plan around Evo2 20B/40B for the first benchmark. The current decision is:

1. Use **Evo2 7B** as the first real foundation-model extractor.
2. Treat Transformer Engine/FP8 as a diagnostic only, not a project requirement.
3. Do not purchase or reserve Hopper hardware only for larger Evo2 models until 7B has been tested against BODA baselines.

The local runtime checker currently reports:

```bash
conda run -n boda_evo2_env python src/foundation/evo2/check_evo2_runtime.py
```

```text
Official local Evo2 OS path: YES
FP8/Transformer Engine stack: NOT READY
Evo2 7B light extraction path: NOT_READY
Cached embedding regression/BODA downstream path: READY
```

Interpretation:

- Transformer Engine is not installed in `boda_evo2_env`, but more importantly the visible GPUs are compute capability 6.0/6.1, not Hopper-class.
- Evo2 7B does not require Transformer Engine, but the official light install still depends on a modern CUDA/Torch/FlashAttention path. FlashAttention 2's CUDA path targets Ampere/Ada/Hopper, while this server has Pascal GPUs.
- This server should remain the BODA/cached-embedding regression host. Real Evo2 7B embeddings should be extracted on a supported GPU host, then copied back as cached `.pt` artifacts.

### Portable host qualification protocol

This plan should be usable on another local machine by a Codex agent. Carry at least these files:

```text
plan/phase1_lib1/learn/evo2_foundation_encoded_seq2expr_plan.md
src/foundation/evo2/check_evo2_runtime.py
src/foundation/evo2/extract_evo2_embeddings.py
```

Ask Codex on the target machine to run:

```bash
python src/foundation/evo2/check_evo2_runtime.py
python src/foundation/evo2/check_evo2_runtime.py --json > evo2_runtime_report.json
```

If the repo is not present, ask Codex to collect the equivalent facts:

```bash
python -V
uname -a || true
sw_vers || true
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits || true
python - <<'PY'
import platform
print("platform", platform.platform())
try:
    import torch
    print("torch", torch.__version__)
    print("cuda runtime", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    print("bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
    print("mps", getattr(torch.backends, "mps", None).is_available() if hasattr(torch.backends, "mps") else None)
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, f"cc={p.major}.{p.minor}", f"vram={p.total_memory/1024**3:.1f} GiB")
except Exception as e:
    print("torch import/status failed:", repr(e))
PY
python - <<'PY'
import importlib.util
for name in ["evo2", "flash_attn", "transformer_engine"]:
    print(name, bool(importlib.util.find_spec(name)))
PY
```

Interpretation rules:

- `READY_TO_TEST`: run the official Evo2 generation test, then a 7 kb embedding smoke test before any full extraction.
- `POSSIBLE_BUT_VRAM_TIGHT`: try `evo2_7b_base`, batch size 1, one layer, one 7 kb sequence. If it OOMs, do not spend more time locally; use a larger GPU or hosted extraction.
- `NOT_SUPPORTED_OFFICIALLY_MACOS_NO_CUDA`: use the machine for downstream cached-embedding heads, notebook analysis, or hosted/NIM API clients, not official local Evo2 extraction.
- `NOT_READY`: missing OS/GPU/package requirements. Do not debug science code until the runtime check improves.

For a promising NVIDIA Linux/WSL2 host, create a separate test environment rather than modifying BODA:

```bash
conda create -n evo2_7b_probe python=3.12 pip -y
conda activate evo2_7b_probe
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install packaging ninja
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
python src/foundation/evo2/check_evo2_runtime.py
python -m evo2.test.test_evo2_generation --model_name evo2_7b_base
```

On Blackwell/RTX 50-series GPUs, if the pinned FlashAttention install fails, ask Codex to re-check the current Evo2 README and FlashAttention package notes before changing pins. Do not modify `boda_env` or the BODA-compatible `boda_evo2_env` to solve Evo2 extraction dependencies.

Then run the smallest direct embedding smoke test:

```bash
python - <<'PY'
import torch
from evo2 import Evo2

model = Evo2("evo2_7b_base")
sequence = "ACGT" * 1750  # 7000 bp
input_ids = torch.tensor(
    model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to("cuda:0")
layer_name = "blocks.28.mlp.l3"
with torch.no_grad():
    outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
print("embedding shape", tuple(embeddings[layer_name].shape))
PY
```

Machine-specific expectations:

- **Apple M3 Max MacBook Pro**: expected to fail official local Evo2 qualification because Evo2's local path is CUDA/NVIDIA, not Apple MPS. It can still run analysis, BODA cached-embedding regressors on CPU/MPS if dependencies install, and hosted API clients.
- **Home desktop with RTX 5080**: promising but not guaranteed. NVIDIA lists the RTX 5080 as Blackwell, CUDA capability 12.0, and 16 GB GDDR7. The architecture is modern enough, but 16 GB VRAM may be tight for Evo2 7B at 7 kb. Use Linux or WSL2, recent NVIDIA drivers, PyTorch CUDA 12.8 wheels, then run the official `evo2_7b_base` generation and one-sequence embedding tests. If 16 GB OOMs, move extraction to a 24-48+ GB NVIDIA GPU.

Source URLs for re-checking on the target machine:

```text
https://github.com/ArcInstitute/evo2
https://pypi.org/project/flash-attn/
https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5080/
https://developer.nvidia.com/cuda-gpus
```

### Is Evo2 7B enough for this task?

For the current biological task, **Evo2 7B is the right first model to test**.

Reasons:

- The full synthetic construct is about **7 kb**, which fits within the `evo2_7b_base` 8K context if the final serialized sequence truly stays below the model limit. If sequence length can exceed 8K after adapters/tags/flanks, use `evo2_7b_262k` or `evo2_7b` rather than moving to 20B/40B.
- The variable regions together are **<2 kb**, so the key supervised signal is likely local motif grammar plus medium-range regulatory context. A 7B genomic foundation model should be a strong enough frozen encoder to test whether foundation embeddings help at all.
- Larger models may improve representation quality, but they are not the first bottleneck. For this benchmark, pooling/reduction strategy, layer choice, exact construct context, target noise, split design, and comparison to BODA baselines are more likely to determine whether the experiment is informative.

Recommended 7B-specific ablation:

1. **Full construct embedding**: all ~7 kb, mean/last/token-reduction over the full sequence.
2. **Variable-window embedding**: extract the same layer but reduce only over annotated variable regions.
3. **Segment-aware embedding**: concatenate reductions over enhancer/promoter/UTR/intron/barcode/terminator or other known EU components.
4. **Variable-only sequence embedding**: <2 kb variable regions stitched in fixed order, used as a lower-cost control.

Decision rule:

- If 7B embeddings beat or complement BODA one-hot baselines on validation/test with identical splits, continue with 7B and improve pooling/head design.
- If 7B embeddings are weak, do not jump straight to 20B/40B. First test layer choice, segment-aware pooling, AlphaGenome/Borzoi-style functional prediction features, and whether the full 7 kb fixed context is washing out the variable-region signal.
- Consider larger Evo2 models only after the 7B pipeline is technically stable and the validation results show a clear reason to spend the extra hardware/runtime.

Recommended upgrade tracks:

1. **Practical 7B embedding extractor**
   - Add one modern high-memory bf16-capable GPU, preferably `L40S 48 GB`, `RTX 6000 Ada 48 GB`, `A100 80 GB`, or `H100/H200`.
   - Use this for `evo2_7b_base`, `evo2_7b`, or `evo2_7b_262k` extraction, starting with short CRE/EU sequences and small batch sizes.
   - This is the cheapest useful local path, but it does not unlock FP8-required 1B/20B/40B models unless the card is Hopper and the FP8 stack passes tests.

2. **Optional Hopper research extractor**
   - Install at least `1x H100 80 GB PCIe` or `1x H200`.
   - Only prioritize this if we decide larger FP8-required models are scientifically necessary after the 7B benchmark.
   - Validate actual model/batch/context combinations with Evo2's generation/embedding tests.

3. **Deferred 40B / NIM-class local serving**
   - Plan for `2x H100 80 GB` or `1x H200 144 GB`, matching NVIDIA's NIM matrix.
   - Add enough local NVMe space for model/container caches and generated embedding artifacts; 1-2 TB free space is more comfortable than the current 190 GB free.
   - Defer this unless 7B/other feature models leave a clear unmet need.

Before purchasing, confirm:

- chassis supports the GPU form factor: passive server GPU versus active workstation GPU;
- PSU capacity and the required PCIe/12VHPWR/EPS auxiliary power cabling;
- airflow is sufficient for 300-700 W accelerator cards;
- BIOS can enumerate the target card(s), ideally with Above 4G Decoding enabled;
- the server can tolerate a driver upgrade to 535+ or preferably a newer production branch;
- existing BODA workflows are preserved by keeping `boda_env` frozen and adding a separate Evo2 extraction environment or container.

Operational rollout:

1. Keep `boda_env` and `boda_evo2_env` unchanged for BODA baselines and cached-embedding regressors.
2. Install the new GPU and update only the host NVIDIA driver first.
3. Verify `nvidia-smi`, thermals, persistence mode, and a small CUDA matmul.
4. Build a separate `evo2_local_env` or Docker/Apptainer image with Python 3.11/3.12, Torch 2.7.x, Flash Attention, optional Transformer Engine, and Evo2.
5. Run upstream Evo2 generation tests before extracting any science embeddings.
6. Run a tiny real extraction on 10-100 CRE/EU sequences, then the Hani 5'UTR layer panel, then in-house Lib1 enhancer/EU sequences.

---

## Key design principles

1. **Frozen first**
   - Do not backpropagate into Evo2 in the first version.
   - Extract embeddings offline.
   - Cache them with metadata.

2. **Identical split comparison**
   - Use the exact same train/validation/test rows as current one-hot BassetVL/ResNet baselines.
   - No row leakage between splits.
   - For barcode-level phase 2, all barcodes from the same construct must remain in the same split.

3. **Train-only preprocessing**
   - Fit target scalers on training rows only.
   - Fit embedding feature scalers on training rows only.
   - Apply frozen train-fit scalers to validation/test.

4. **Layer probing before large HPO**
   - First search layer choice cheaply using a simple fixed supervised head.
   - Then train MLP heads only for top layer candidates.

5. **Held-out evaluation**
   - Use validation to choose layer/head/hyperparameters.
   - Use test only for final reporting.

6. **Compare to sequence baselines**
   - Evo2 embedding models must be compared against current one-hot BassetVL/ResNet models on identical splits and targets.

---

## Phase 0: Define expected data table

The embedding extraction script should produce or consume a table with at least:

```text
construct_id
sequence
split                # train / val / test
RNA_target            # optional
protein_target        # optional
multi_output_targets  # optional, can be multiple columns
n_barcodes            # optional, for weighting
source_dataset
source_row_index
```

For the full Expression Unit setting, `sequence` should be the full **synthetic DNA construct**, intended length around **7 kb**. The known variable regions should also be annotated because their combined length is usually **<2 kb**, and variable-region or segment-aware pooling may be more informative than whole-sequence mean pooling.

For current lib1 enhancer testing, the pipeline can still support the existing shorter padded/flanked sequence representation, but the code should not hard-code 600 bp assumptions into the embedding model family.

Recommended CLI args:

```bash
--sequence_column sequence
--id_column construct_id
--target_columns RNA_target protein_target
--split_column split
--input_len 7000
```

---

## Phase 1: Evo2 embedding extraction

### Goal

Create a script that loads synthetic DNA constructs, runs Evo2 in frozen inference mode, extracts candidate hidden layers, and saves construct-level embeddings.

Suggested script:

```text
src/foundation/evo2/extract_evo2_embeddings.py
```

Suggested output directory:

```text
src/learn/local_artifacts/foundation_embeddings/evo2/
```

### Candidate layer search

Start with a sparse layer panel rather than every layer:

```python
candidate_layers = [
    "blocks.8",
    "blocks.16",
    "blocks.28",
    "blocks.L"   # final layer; replace with exact Evo2 module name
]
```

The exact layer names should match the Evo2 API. If the Evo2 repo exposes names like `blocks.28.mlp.l3`, support that exact string as a CLI value.

Recommended CLI:

```bash
python src/foundation/evo2/extract_evo2_embeddings.py \
  --input_table data/processed/synthetic_constructs.csv \
  --sequence_column sequence \
  --id_column construct_id \
  --split_column split \
  --target_columns RNA_target protein_target \
  --model_name evo2_7b \
  --layers blocks.8 blocks.16 blocks.28 blocks.28.mlp.l3 final \
  --output_dir src/learn/local_artifacts/foundation_embeddings/evo2/lib1_or_EU_v1
```

Plumbing-only smoke example that does not require Evo2 or a GPU:

```bash
python src/foundation/evo2/extract_evo2_embeddings.py \
  --input_table data/processed/synthetic_constructs.csv \
  --sequence_column sequence \
  --id_column construct_id \
  --split_column split \
  --target_columns RNA_target protein_target \
  --model_name hash_smoke \
  --layers blocks.8 blocks.16 final \
  --backend hash_smoke \
  --hash_smoke_dim 64 \
  --smoke_n 8 \
  --output_dir src/learn/local_artifacts/foundation_embeddings/evo2/hash_smoke_plumbing
```

The `hash_smoke` backend is only for validating row order, artifact format, datamodule loading, and trainer wiring. It must not be used for benchmark results.

### Embedding output format

Prefer one artifact per layer, plus a manifest.

```text
lib1_or_EU_v1/
  manifest.json
  rows.parquet
  embeddings__model-evo2_7b__layer-blocks_8.pt
  embeddings__model-evo2_7b__layer-blocks_16.pt
  embeddings__model-evo2_7b__layer-blocks_28.pt
  embeddings__model-evo2_7b__layer-final.pt
```

`rows.parquet` should contain row-level metadata:

```text
construct_id
source_row_index
sequence_length
split
target columns
n_barcodes if available
sequence_hash
```

Each `.pt` file should contain:

```python
{
    "embedding": torch.Tensor,      # shape [n_constructs, d]
    "construct_id": list[str],
    "layer_name": str,
    "model_name": str,
    "sequence_hash": list[str],
    "pooling_or_reduction": str,
    "created_at": str,
    "evo2_commit_or_version": str,
}
```

Even if we avoid saying “pooling” in figures, the implementation still needs to reduce token-level hidden states to one construct-level vector. Keep this detail in code metadata.

Recommended default reduction:

```text
mean over sequence positions
```

But the public-facing figure can simply label the result as:

> Sequence embedding

### Smoke tests

Run embedding extraction on 8 constructs and verify:

- deterministic row order;
- each output layer has the same construct order;
- shape is `[8, d]`;
- no NaNs or infs;
- layer metadata is saved;
- split labels and targets match input table.

---

## Phase 2: Embedding data module

### Goal

Add a data module that loads cached Evo2 embeddings and returns embedding-target pairs for supervised training.

Suggested file:

```text
boda/data/embedding_datamodule.py
```

Suggested class:

```python
class EmbeddingRegressionDataModule(pl.LightningDataModule):
    ...
```

### Required features

- Load `rows.parquet` or fallback `rows.csv` and one selected embedding `.pt` file.
- Select rows by `split` column.
- Select one or more target columns.
- Fit feature scaler on train embeddings only.
- Fit target scaler on train targets only, unless targets were already explicitly standardized upstream.
- Return `(embedding, target)` for unweighted training.
- Return `(embedding, target, weight)` if `--use_weights true`.

### Suggested CLI args

```bash
--embedding_dir src/learn/local_artifacts/foundation_embeddings/evo2/lib1_or_EU_v1
--embedding_file embeddings__model-evo2_7b__layer-blocks_28.pt
--rows_file rows.parquet
--target_columns RNA_target protein_target
--split_column split
--batch_size 128
--standardize_x true
--standardize_y true
--use_weights false
--weight_column n_barcodes
--min_weight 0.1
--b_cap 10
```

### Barcode-based weights

For construct `i` with barcode count `b_i`:

```text
raw_weight_i = log(1 + b_i) / log(1 + b_cap)
w_i = max(min_weight, min(1.0, raw_weight_i))
```

Weighted MSE:

```text
L = sum_i w_i * (yhat_i - y_i)^2 / max(sum_i w_i, 1e-8)
```

This should only be used when `n_barcodes` is available and intentionally enabled.

---

## Phase 3: Trainable heads

### 3A. MLP regressor

Suggested file:

```text
boda/model/embedding_heads.py
```

Suggested class:

```python
class EmbeddingMLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_outputs: int = 1,
        hidden_dim: int = 512,
        n_hidden_layers: int = 2,
        dropout_p: float = 0.1,
        activation: str = "GELU",
        loss_criterion: str = "MSELoss",
    ):
        ...
```

Recommended first MLP grid:

```text
hidden_dim: 256, 512, 1024
n_hidden_layers: 1, 2
learning_rate: 1e-4, 3e-4, 1e-3
dropout_p: 0.0, 0.1, 0.2
batch_size: 64, 128
```

### 3B. Heteroscedastic head

Use this only after the basic MLP is working.

The heteroscedastic head predicts:

```text
mu(x)       = predicted expression mean
log_var(x)  = predicted log variance
```

Gaussian NLL:

```text
L_i = 0.5 * ( exp(-log_var_i) * (y_i - mu_i)^2 + log_var_i )
```

For multi-output RNA/protein, predict one mean and one log variance per output:

```text
output_dim = 2 * n_targets
```

Recommended class:

```python
class EmbeddingHeteroscedasticRegressor(nn.Module):
    ...
```

Clamp `log_var` for numerical stability:

```python
log_var = torch.clamp(log_var, min=-10.0, max=5.0)
```

---

## Phase 4: Embedding training graph

The existing `CNNBasicTraining` graph assumes every batch is `(x, y)` and applies `self.criterion(y_hat, y)`. Add a separate graph to avoid breaking CNN training.

Suggested file:

```text
boda/graph/embedding_prediction.py
```

Suggested classes:

```python
class EmbeddingRegressionTraining(LightningModule):
    ...

class WeightedEmbeddingRegressionTraining(EmbeddingRegressionTraining):
    ...

class HeteroscedasticEmbeddingRegressionTraining(LightningModule):
    ...
```

### Required metrics

Log these for train/val/test where possible:

```text
loss
MSE
standardized_MSE
Pearson r
Pearson r^2
COD R^2
Spearman r
```

For multi-output targets:

```text
val_pearson_mean
val_pearson_RNA
val_pearson_protein
val_cod_r2_mean
val_cod_r2_RNA
val_cod_r2_protein
```

Use validation metrics for checkpointing.

Recommended checkpoint metric:

```text
val_pearson_mean
```

or, for single-output:

```text
val_pearson
```

Use `mode=max` for correlation/R² metrics.

---

## Phase 5: Layer probing workflow

### Stage 1: cheap layer screen

For each candidate layer:

1. Load embeddings from that layer.
2. Train the same simple MLP or a very small fixed head.
3. Evaluate validation Pearson and COD R².
4. Rank layers.

Recommended first fixed head:

```text
hidden_dim = 256
n_hidden_layers = 1
dropout_p = 0.1
lr = 3e-4
max_epochs = 50
early_stopping_patience = 10
```

Output summary:

```text
layer_probe_summary.csv
```

Columns:

```text
model_name
layer_name
embedding_dim
n_train
n_val
n_test
target_columns
val_loss
val_pearson_mean
val_cod_r2_mean
val_spearman_mean
test metrics only if this is final selected config
```

Important:

> During layer probing, do not use test metrics to choose the layer.

### Stage 2: refine top layers

Take top 2–3 layer candidates from validation performance and run a small MLP HPO.

### Stage 3: final test reporting

Only after selecting:

- layer;
- target columns;
- head type;
- MLP hyperparameters;
- weighting on/off;

run final test reporting and compare with one-hot baselines.

---

## Phase 6: Baselines and fairness checks

### Required baselines

1. Current one-hot BassetVL or ResNet model on identical rows.
2. Simple sequence-feature baseline, if easy:
   - GC content;
   - k-mer ridge;
   - length / design metadata controls if applicable.
3. Evo2 embedding MLP.
4. Evo2 embedding heteroscedastic head, optional phase 2/3.

### Fairness rules

- Same train/val/test split.
- Same target columns.
- Same target normalization rules.
- Same excluded rows.
- Same held-out test reporting.
- Same barcode thresholding policy.

---

## Phase 7: Replicate-aware / barcode-aware extension

### Phase 7A: weighted sequence-level training

Use sequence-level target table with `n_barcodes` as reliability.

Batch format:

```python
embedding, target, weight
```

Loss:

```text
weighted MSE
```

### Phase 7B: raw barcode-level training

If raw barcode rows are available, create a replicate-aware dataset:

```text
construct_id
barcode_id
sequence
RNA_measurement
protein_measurement
split
```

Rules:

- One embedding per construct.
- Multiple barcode observations can point to the same construct embedding.
- All barcodes from a construct must remain in the same split.
- Train heteroscedastic head to distinguish predictable biological signal from measurement noise.

---

## Suggested implementation tasks for Codex

### Task 1: Create embedding extraction skeleton

Create:

```text
src/foundation/evo2/extract_evo2_embeddings.py
```

Status: implemented as a scaffold with `--backend evo2` for real extraction and `--backend hash_smoke` for deterministic plumbing checks.

Requirements:

- CLI parser.
- Load input CSV/parquet.
- Validate sequence column and target columns.
- Hash sequences.
- Load Evo2 in frozen inference mode.
- Extract requested layers.
- Save one `.pt` per layer and a manifest.
- Add a `--smoke_n 8` option.

If Evo2 is not installed in the current environment, write the script with clear import errors and environment instructions, but keep the rest importable.

### Task 2: Add embedding data module

Create:

```text
boda/data/embedding_datamodule.py
```

Status: implemented and registered as `boda.data.EmbeddingRegressionDataModule`.

Requirements:

- Load rows and embeddings.
- Align by construct_id and row order.
- Split by split column.
- Train-only x/y scaling.
- Optional weight calculation.
- Unit tests for no split overlap.

### Task 3: Add embedding heads

Create:

```text
boda/model/embedding_heads.py
```

Status: implemented and registered as `EmbeddingMLPRegressor` and `EmbeddingHeteroscedasticRegressor`.

Implement:

- `EmbeddingMLPRegressor`
- `EmbeddingHeteroscedasticRegressor`

### Task 4: Add embedding training graphs

Create:

```text
boda/graph/embedding_prediction.py
```

Status: implemented and registered as `EmbeddingRegressionTraining`, `WeightedEmbeddingRegressionTraining`, and `HeteroscedasticEmbeddingRegressionTraining`.

Implement:

- unweighted MSE training;
- weighted MSE training;
- heteroscedastic Gaussian NLL training;
- single-output and multi-output metrics.

### Task 5: Add layer probing runner

Create:

```text
src/foundation/evo2/run_layer_probe.py
```

or:

```text
src/learn/evo2/run_evo2_layer_probe.py
```

Status: implemented at `src/learn/evo2/run_evo2_layer_probe.py`.

Requirements:

- Iterate over embedding files/layers.
- Run fixed-head training.
- Collect validation metrics.
- Save `layer_probe_summary.csv`.
- Do not use test metrics for selection by default.

### Task 6: Add launch scripts

Create examples:

```text
scripts/evo2/01_extract_embeddings_smoke.sh
scripts/evo2/02_extract_embeddings_full.sh
scripts/evo2/03_layer_probe.sh
scripts/evo2/04_train_best_mlp.sh
scripts/evo2/05_train_weighted_mlp.sh
scripts/evo2/06_train_heteroscedastic_head.sh
```

Status: pending. The Python entry points now exist, but shell launch wrappers should wait until the target EU/lib1 input table path and the real Evo2 environment are pinned down.

---

## Suggested CLI examples

### Extract embeddings

```bash
python src/foundation/evo2/extract_evo2_embeddings.py \
  --input_table data/processed/synthetic_constructs.csv \
  --sequence_column sequence \
  --id_column construct_id \
  --split_column split \
  --target_columns RNA_target protein_target \
  --model_name evo2_7b \
  --layers blocks.8 blocks.16 blocks.28 blocks.28.mlp.l3 final \
  --output_dir src/learn/local_artifacts/foundation_embeddings/evo2/EU_7kb_v1
```

### Layer probing

```bash
python src/learn/evo2/run_evo2_layer_probe.py \
  --embedding_dir src/learn/local_artifacts/foundation_embeddings/evo2/EU_7kb_v1 \
  --target_columns RNA_target protein_target \
  --hidden_dim 256 \
  --n_hidden_layers 1 \
  --dropout_p 0.1 \
  --lr 3e-4 \
  --max_epochs 50 \
  --checkpoint_monitor val_pearson_mean \
  --stopping_mode max \
  --output_csv results/evo2_layer_probe_summary.csv
```

### Train best MLP

```bash
python src/learn/train.py \
  --data_module EmbeddingRegressionDataModule \
  --model_module EmbeddingMLPRegressor \
  --graph_module EmbeddingRegressionTraining \
  --embedding_dir src/learn/local_artifacts/foundation_embeddings/evo2/EU_7kb_v1 \
  --embedding_file embeddings__model-evo2_7b__layer-blocks_28.pt \
  --target_columns RNA_target protein_target \
  --standardize_x true \
  --standardize_y true \
  --n_outputs 2 \
  --hidden_dim 512 \
  --n_hidden_layers 2 \
  --dropout_p 0.1 \
  --optimizer AdamW \
  --lr 3e-4 \
  --checkpoint_monitor val_pearson_mean \
  --stopping_mode max
```

Exact optimizer CLI may need to match the existing BODA optimizer arg parser.

If the extraction environment lacks a parquet engine, the extractor writes `rows.csv`; pass `--rows_file rows.csv` to the data module or layer-probe runner in that case.

---

## Unit tests

Add tests under:

```text
tests/test_embedding_datamodule.py
tests/test_embedding_heads.py
tests/test_embedding_training.py
tests/test_evo2_embedding_manifest.py
```

Current status: a first smoke test file exists at `tests/test_embedding_pipeline_smoke.py`. It covers the cached embedding data module, MLP/heteroscedastic output shapes, weighted MSE, and `hash_smoke` extraction artifacts. It could not be run in the current shell because `torch` and `pytest` are missing from the active Python environment.

Minimum tests:

1. Embedding rows align with `rows.parquet`.
2. No construct overlap between train/val/test.
3. Feature scaler is fit only on train rows.
4. Target scaler is fit only on train rows.
5. Weighted dataloader returns `(x, y, w)`.
6. Unweighted dataloader returns `(x, y)`.
7. MLP output shape equals `[batch_size, n_outputs]`.
8. Heteroscedastic output shape equals `[batch_size, 2 * n_outputs]`.
9. Weighted MSE matches hand-calculated formula on a toy batch.
10. Gaussian NLL is finite on a toy batch.

---

## Acceptance criteria

The first complete implementation is successful if:

- Evo2 embeddings can be extracted for a smoke-test set.
- Multiple candidate layers can be cached and loaded.
- `EmbeddingRegressionDataModule` trains an MLP on one selected layer.
- A layer-probing summary ranks candidate layers by validation performance.
- Final selected Evo2 embedding model is compared to one-hot BassetVL/ResNet on the same split.
- Test metrics are reported only after model/layer selection.
- All artifacts contain enough metadata to reconstruct the run.

---

## Reporting template

For each final run, report:

```text
Dataset: EU or lib1 enhancer
Input: synthetic DNA construct, length ___ bp
Foundation model: Evo2 ___
Frozen or fine-tuned: frozen
Embedding layer: ___
Embedding dimension: ___
Trainable head: MLP / heteroscedastic head
Targets: RNA / protein / RNA+protein
Train/val/test n: ___ / ___ / ___
Target normalization: train-only z-score yes/no
Feature normalization: train-only z-score yes/no
Weighting: none / barcode-weighted
Validation selection metric: ___
Test Pearson: ___
Test COD R²: ___
Test Spearman: ___
One-hot baseline test Pearson: ___
One-hot baseline test COD R²: ___
```

---

## Notes for future extension

After the frozen-embedding benchmark is stable, possible extensions are:

1. **Fine-tune only a small adapter on top of Evo2 hidden states**
   - LoRA/adapters if the environment supports it.

2. **Layer ensembling**
   - Only after individual layer probing is understood.
   - Combine top layers by concatenation or learned weighted average.

3. **Joint RNA/protein modeling**
   - Shared MLP trunk with separate output heads.
   - Compare to separate RNA-only and protein-only heads.

4. **Uncertainty-aware active learning**
   - Use heteroscedastic variance and/or ensemble disagreement for acquisition.

5. **Construct component attribution**
   - After predictor works, evaluate whether embeddings/heads capture enhancer, promoter, UTR, intron, exon, barcode, and terminator contributions.

---

## Recommended first milestone

Do not start with full HPO. Start with:

```text
1 dataset
4 Evo2 layers
1 target or 2 targets
1 fixed MLP head
3 random seeds
same split as one-hot baseline
```

Then decide whether Evo2 embeddings provide enough signal to justify a larger sweep.
