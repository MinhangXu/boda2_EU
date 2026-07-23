# Lib1 Dedup Stage 3 Weighted-Loss Protocol Amendment

**Freeze date:** 2026-07-14
**Time basis:** America/Chicago (UTC file timestamps may show 2026-07-15)
**Status:** frozen implementation contract; dry-run manifest independently
validated; no Stage 3 cell launched
**Audit status:** loader not instantiated, audit IDs not materialized, audit
stratum counts not inspected

## Decision And Scope

This amendment is the binding Stage 3 entry contract following the completed
Stage 2 screen and bounded targeted 3'UTR HPO. It supersedes every earlier
provisional reference to five configurations per part, a 250-cell weighted
campaign, a global one-SE/simple-model rule, or a global RMSE/COD tolerance.

Stage 3 asks one controlled question: for each already-completed unweighted
development cell, does the frozen barcode-weighted training loss improve the
same configuration, fold, RC condition, model seed, data product, and split?
It is not another architecture HPO. Each part therefore has a predeclared
ten-configuration portfolio. All ten are eligible for final selection.

This amendment authorizes implementation, manifest generation, static
validation, and representative one-row pilots. It does **not** authorize the
full campaign or any audit/test-loader construction.

## Frozen Portfolios

The full `base_config_id` is the identity. Rank is portfolio order, not a new
performance claim. The machine-readable portfolio is an additional exact
freeze and must agree byte-for-byte with this list.

### Enhancer: six transfer routes and four scratch route controls

1. `basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036` — K562, full transfer
2. `basecfg_e53d6596a16e9f43bfe71e4ea2a364dd30237733beee9030030ecbc84f6d30a0` — HepG2, full transfer
3. `basecfg_3f7d963d6d647ee5eb5ee02239f1b0c992c3f33d90200d52b4e00c88e7ddd02d` — HepG2, conv3-plus transfer
4. `basecfg_f199d009d69405a41890a39cf91759eeb6c27df03f0082200b4505f78918b82b` — K562, conv3-plus transfer
5. `basecfg_404c9e99e7e9571266e83c07b5a5016a731b52212ba3723d98a6d0b44b378cec` — K562, branched-only transfer
6. `basecfg_d7ab0bf6f1bc39af9c4ff9269d2ed0e47f5720b1933057b9654a388f5ed0422f` — HepG2, branched-only transfer
7. `basecfg_5d9f63c25515a73921372a308950d2d79367da2c06e35899575ff2b88c000b5e` — scratch anchor 1
8. `basecfg_18119f07e851868812804e4fd3e36585fa0e472b47e71c913886e7ebba668bd9` — scratch anchor 2
9. `basecfg_7bb5763f52f3678922d64e5026e75fa14b79bde606319b207a5f8b30885f87b8` — scratch anchor 3
10. `basecfg_246106d4d9907232c48b9d670cb58642ec84b1ac712d4ce21636ff0d33a81c18` — scratch anchor 4

The scratch configurations are scientifically useful route controls, but
they are not quarantined anchors: if they pass the same gates and selection
rule, they may become the final Enhancer model.

### Promoter

1. `basecfg_00175f1ce3e6b9bb7d49b89360083a7314cb294edf173d05d9f076913387c74e`
2. `basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d`
3. `basecfg_e10d0e2bdadc81888c0cd24f22194f01c9bb752fb97bcad66b6fb20da5fe66eb`
4. `basecfg_0c0cefe749c9241f03f893c1fcfe585418a91d151686ee2d4b0eca54335790f8`
5. `basecfg_9b9293193ecdac4bffee9b00e58cfdde742789ac1c2d1d625047d4578e4fc5fe`
6. `basecfg_f3fa8318ff61c1cb8758134e7dbb9ad2640bdb358c2a736c5d96495080105c4d`
7. `basecfg_9821907e1ab3069b1657e66e9befa92e967038385a0909eb1bda10b1d2df24d0`
8. `basecfg_badc6370f710b1ac55fcd2d4d6de22daa862d46aac6c273fa25fdb638bb8f46c`
9. `basecfg_fe3d6b7e556cd3237d8f537038331229d2a708f32de8d5391056bb5b02ac16f0`
10. `basecfg_408bbe2f201458b3b2c75f768501e9cc824bbcc62f120911829322422bea82cb`

### Intron

1. `basecfg_6079cd38f32d3f5cf024c66fb43e7f88c2ced932f984fbebe30ba99672641b74`
2. `basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86`
3. `basecfg_0ee9e54c8bfb2917566afaa790fcd981a007bb4c35d8427b5a83ba69335c08f3`
4. `basecfg_873605b1a4643a9a8745b10c68faf5f3d485637b9677805410b172f71af146f1`
5. `basecfg_767a6d28b3510037a8510a0d41e00df9106064b13af76975e225bd0e8bcb94d7`
6. `basecfg_710db0cc09f3c386b726a49fd23be30e0fea4896711404a393d54b60d945de4f`
7. `basecfg_a76fef1421c97368714a0ae354db301f69a9a6a7561244523b4106a65fe4a093`
8. `basecfg_5b5d2d82cef98c6e0c7522dbbc388ef4da59ee65687f40159e7c9548eb2277f3`
9. `basecfg_0c59aba4ea114b651fae8352b2a9a3f9010edbd17fe288b902d80ba25c2a0223`
10. `basecfg_e3b7fc22d2bedc66c15d3e7ce8aaaa44679a8305e1719a85c3f8bb51dcb508ca`

### 3'UTR: seven UTRBassetVL configurations and three ResNet1D anchors

1. `basecfg_6cb459958ae1a16e112bdacc6e03c9e02fc12cdc85ed951cfcd25ada7856a517`
2. `basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe`
3. `basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062`
4. `basecfg_8b14e9e7f2f26e52985dda2dec8f128c9da9a31662a64015dca76a993b4cd5b4`
5. `basecfg_1becdea28bb6a22dbb61a48222baf1cbce413ac6e405691c9bda4b1da6253f90`
6. `basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05`
7. `basecfg_1e3a0c9f053271a63a4da596c588484b52c56cf65fe6fb791bd909e15c3b9def`
8. `basecfg_ec031204c44d76ed859477d8b2fcb74f54daf5a9d6d70017728dac5dcabbeb2b`
9. `basecfg_585fba9a4fec47048843b484fad428e9a5236fffe3aec370c3938fb4db39fa92`
10. `basecfg_231fe76767cea395f9dc5ae2625155780ac85b83d944e3ce97ba494417a21fd7`

3'UTR is RC-off only in Stage 3. The three ResNet1D members preserve a bounded
architecture comparison; selecting a ten-member portfolio does not imply an
ensemble and does not require selecting ResNet1D.

### 5'UTR

1. `basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315`
2. `basecfg_25d3b0fb122d4da050145825875c04f5cedc047178b5d2d159d2275a5731f227`
3. `basecfg_e3b85c86fe400906280db9093b388bb1b74a552467120eac98e86c5202650d17`
4. `basecfg_99b40ac8bca80e76b56403be8b15214c10cf6fc33730d7dd3926997792fef16b`
5. `basecfg_bee0f2b508e0fbc529890aafd7b63c93a4014e7bee8ecd46f99e9ddb5481be5f`
6. `basecfg_c9a37b4a162fd8fefbde5b01aaf7556931ec254ec2f1abefa2c9b0f4becb4b56`
7. `basecfg_ffd4992641df6d33f2b23c1aa5857ceab29a6ae247d489d220b53177871f1369`
8. `basecfg_65e011f225d06cf57c83ac305545a839271748462292e326ce22262b13c5fe94`
9. `basecfg_2106736d06b1570dbc9725701e675122292ada6893680b141ece9a9c7a79e82b`
10. `basecfg_d5ad87bb22a68b1d8dd7d91351fafcb8f2d38ac7b7d3f40bc17947d4b9a28be8`

## Frozen Factorial And Cell Accounting

An **arm** is one `(part, base_config_id, loss_mode, rc_mode)` condition pooled
over its five held-out development folds. A **cell** is one arm-fold training
run. The only new Stage 3 factor is weighted versus the immutable unweighted
mate; RC remains a factor for four parts and is fixed off for 3'UTR.

| Part | Configs | RC modes | New weighted cells | Reused unweighted cells | Analysis cells |
|---|---:|---:|---:|---:|---:|
| Enhancer | 10 | off, on | 100 | 100 | 200 |
| Promoter | 10 | off, on | 100 | 100 | 200 |
| Intron | 10 | off, on | 100 | 100 | 200 |
| 3'UTR | 10 | off only | 50 | 50 | 100 |
| 5'UTR | 10 | off, on | 100 | 100 | 200 |
| **Total** | **50** |  | **450** | **450** | **900** |

This yields 180 complete OOF arms, 450 fold-level weighted/unweighted loss
pairs, 400 fold-level RC pairs, 40 complete config-level two-by-two RC-by-loss
factorials, and 80 config-level same-loss RC arm contrasts.

For each of the 40 complete factorials, report the descriptive interaction
`(weighted RC-on - weighted RC-off) - (unweighted RC-on - unweighted RC-off)`.
This difference-in-differences is diagnostic only and cannot enter a gate or
selection.

For capacity planning, summing the observed fit times of the 450 exact source
cells gives a 43.65 GPU-hour replay proxy (median 274 seconds per cell). Eight
equivalent GPUs imply an idealized lower bound of about 5.5 wall-clock hours,
before scheduling and I/O overhead. This is a planning estimate, not a runtime
or quota guarantee; the runner still requires a fresh launch-time GPU/storage
preflight. Model artifact retention remains `none` for all 450 new cells.

## Frozen Weighted-Loss Contract

For sample `i`, with barcode count `n_i`,

`w_i = clip(log1p(n_i) / log1p(8), 0.1, 1.0)`

and training minimizes

`sum_i(w_i * mean_j((prediction_ij - target_ij)^2)) / sum_i(w_i)`.

Weights are required during training; missing, non-finite, negative,
wrong-length, or zero-sum weights are fatal. Validation predictions and all
OOF metrics remain unweighted. Scratch routes use
`CNNWeightedRegressionTraining`; scoped Enhancer transfer routes use
`CNNBassetBranchedScopedWeightedTransfer`. The latter must preserve the exact
source-head slice, warm-up/final transfer scope, differential learning rates,
and optimizer-state reset while consuming the weight tensor.

## Frozen Development Gates And Selection

Every weighted arm is compared only with its exact unweighted mate. The
primary weighted-loss gate requires all of the following:

- mean of five paired fold-Pearson deltas at least `+0.005`;
- positive Pearson delta in at least four of five folds;
- pooled OOF RMSE increase no larger than the part-specific margin below;
- pooled OOF COD R2 decrease no larger than the paired part-specific margin;
- for Intron, mean paired within-inferred-stratum-centered Pearson delta at
  least zero, with no more than two negative folds. This Intron centered
  criterion applies independently to both loss gates and RC gates; an RC-on
  weighted arm must satisfy it for both of its intervention comparisons.

Each RMSE margin is exactly one percent of the median pooled OOF RMSE across
the frozen unweighted portfolio arms for that part. The COD allowance is the
RMSE-equivalent degradation on the same part's OOF target-variance scale.
These are decision guardrails, not quantities used to train a model and not
post-audit thresholds.

| Part | Reference median RMSE | Allowed RMSE increase | Allowed COD R2 decrease |
|---|---:|---:|---:|
| Enhancer | 0.7880613432463801 | 0.0078806134324638 | 0.015487742323554538 |
| Promoter | 0.4370511371118717 | 0.004370511371118717 | 0.016236508146719106 |
| Intron | 0.5381529502263820 | 0.00538152950226382 | 0.01244647116447774 |
| 3'UTR | 0.6338276349695062 | 0.006338276349695062 | 0.01682952454591347 |
| 5'UTR | 0.38481036609798924 | 0.0038481036609798926 | 0.016323993067143768 |

For parts with both RC modes, RC-on must pass the analogous paired gate over
the same-loss RC-off arm; 3'UTR has no RC decision. A complete RC-off
unweighted arm with finite required selection metrics is the admissible
baseline. A weighted arm becomes admissible
only after its loss gate passes; an RC-on arm becomes admissible only after its
RC gate passes; an RC-on weighted arm must pass both. Selection is performed
independently per part among admissible arms:

Non-finite values fail closed. Any non-finite required pooled or five-fold
input makes the corresponding intervention gate fail. Any arm with non-finite
pooled Pearson, RMSE, COD R2, or minimum-fold Pearson is ineligible; Intron
also requires finite minimum-stratum and within-stratum-centered Pearson. Any
non-finite best-arm bootstrap replicate is a fatal analysis-contract error.
Undefined descriptive calibration quantities are retained with an explicit
reason but do not alone determine admissibility.

1. find the arm with the highest pooled five-fold raw-scale OOF Pearson; an
   exact best-point tie uses the same downstream deterministic ordering below
   to choose the bootstrap reference;
2. independently for each part, reinitialize
   `numpy.random.default_rng(20260714)`, resample held-out rows with replacement separately
   within each fold, concatenate the five resampled folds, and recompute pooled
   Pearson 10,000 times for that best arm; its bootstrap sample standard
   deviation is `SE_best`, and the one-SE set contains every admissible arm
   whose point-estimate pooled Pearson is at least
   `Pearson_best - SE_best`;
3. within that band, prefer highest minimum-fold Pearson; for Intron next
   prefer highest minimum inferred-stratum Pearson and then highest pooled
   within-inferred-stratum-centered Pearson; then prefer lower pooled RMSE and
   higher pooled COD R2;
4. exact metric ties only: if the tied block contains only Enhancer transfer
   routes, prefer the narrower scope (`branched_only`, then `conv3_plus`, then
   `full`); if it contains any scratch route, or belongs to another part,
   prefer fewer total parameters. Resolve a residual equal-parameter
   Enhancer-transfer tie by narrower scope, then prefer RC off, then
   unweighted loss, then the lexicographically smaller full `base_config_id`.

There is no global five-slot fill, no global simplicity score, and no rule
that forces architecture diversity. This is a part-specific selection over
the frozen portfolios.

## Intron Development Sensitivity And Final Audit

The collaborator-provided labels available in the current data are three
nested sequence masks, not verified synthesis-pool membership. They are
assigned deterministically with precedence and named:

1. `mask1_specific`;
2. `mask2_not_mask1`;
3. `mask3_residual`.

Every Intron Stage 3 OOF arm must report natural-mixture pooled metrics,
within-stratum-centered metrics, macro and minimum-stratum metrics,
per-stratum calibration, the fold-trained stratum-mean baseline, and an
equal-stratum sensitivity estimate with effective sample size. Minimum-stratum
and within-stratum-centered Pearson enter the Intron ordering exactly where
specified above; the remaining diagnostics are reporting checks only. None may
change the frozen rows or return a model to HPO.

The canonical development-only fold-trained stratum-mean baseline prediction
product is frozen at SHA-256
`82c228a3ba0cd0b0df403b52095f8efc1a9a3cdd20417a656b8cccb8f2d14e8c`;
Stage 3 analysis must reject a changed file rather than silently recompute or
substitute it.

Only after all Stage 3 development decisions are irreversibly frozen may the
existing natural 265-row Intron audit be opened once. The same predictions
must receive natural-mixture and inferred-mask-stratum reporting, with an
optional equal-stratum sensitivity estimate. Barcode-cutoff summaries are
predeclared at 8, 10, and 12; always show `n`, and suppress category Pearson
when `n < 30` or either variable has zero variance. No balanced replacement
audit subset is permitted.

## W&B Organization And Audit Isolation

Entity: `minhangxu1998-baylor-college-of-medicine`.

Each part uses
`<part>__bashor_in_house__dedup_exact_v1__stage3_weighted_development`.
Groups bind part, full config identity, and weighted/unweighted pairing;
`cell_id`, `loss_pair_id`, `source_unweighted_cell_id`, fold, RC, data/split
hashes, graph class, and exact run name are required provenance.

The generator, verifier, and runner are development-manifest-only programs.
They must keep `evaluate_test_after_fit=false`, epoch metrics at `train val`,
predictions at `val`, `n_test=0`, and must reject audit/test options. They do
not import a DataModule, instantiate any loader, enumerate audit IDs, or
inspect audit stratum counts. A training provenance record may retain the
split manifest's precomputed audit-exclusion hash so registry and provenance
can prove separation; that hash is not an audit loader, prediction, metric, ID
listing, or authorization to inspect audit composition.

## Frozen Implementation Products And Pilot Gate

- Portfolio SHA-256: `8716a2fe0c6e30bb54a925555cca312f69c1608dd2828116b0ae71ba4fc06bf3`
- Weighted dry-run manifest SHA-256: `09de6182cf107c7b9485390fc9556ac48a92efe776bc35ab3ea6ca01a0ebca44`
- Analysis manifest SHA-256: `7b2d4115e697b8ac9507b3a8e1f5ce22aa55a6da8c2fb826d9b52992932d5995`
- Reused-unweighted manifest SHA-256: `648b2f6c7fd3fa905eda2e3f13817b83c0827ab63d358d45b8783f0c29b078c7`
- Static validation status: passed, 0 commands executed
- Completion/full-analysis program:
  `src/analysis/lib1_dedup_stage3_analysis.py`; its default full path requires
  all 900 cells and its `--readiness-only` path cannot perform selection
- Interpretation notebook:
  `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/05_stage3_paired_rc_loss_analysis.ipynb`;
  it consumes analyzer products and contains no gate or audit-loader logic

Before a full launch is separately authorized, run and verify these exact
pilots in order:

1. manifest row 1, `cell_fd3ec0f68e4c3b375e52`: Enhancer K562/full scoped
   transfer, fold 0, RC off;
2. manifest row 61, `cell_b448ff21b125f3e8cfc0`: Enhancer scratch anchor 1,
   fold 0, RC off.

Each pilot must finish with exactly one **completed** matching registry row,
one validation-prediction file, one provenance record, `n_test=0`, the expected
held-out-row hash/count recomputed from the prediction IDs, and the expected
pairing identities. Failed-attempt rows remain as provenance and do not block
a retry, but a second completed row is forbidden. The runner must
verify both records and record fresh GPU, storage, and W&B preflight evidence
before accepting a separately confirmed non-pilot launch. Pilot success does
not itself authorize the remaining 448 cells.
