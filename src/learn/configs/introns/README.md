# Introns HPO Configs

This directory now contains runnable intron configs plus the original
placeholder template.

## Bashor In-House Lib1 Intron

The first in-house intron scratch baseline uses:

- source table:
  `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/introns/L1_final_fastqs1-5_sublibrary_Intron_subset.csv`
- prep script:
  `src/learn/prepare_lib1_intron_inhouse_dataset.py`
- learn-ready table:
  `src/learn/derived_data/introns/bashor_in_house/lib1_intron_modal80_fastqs1_5__learn_ready.tsv`
- config:
  `introns/bashor_in_house/resnet1d/lib1_intron_modal80__scratch_resnet1d__bayes.yml`
- launcher:
  `src/learn/launch/lib1_intron_scratch_resnet1d_sweep.sh`

The prep script keeps valid finite-positive rows at the modal 80 nt intron
length by default. The sweep uses `Lib1IntronDataModule` with high-quality
heldout rows defined by `n_barcodes >= 8`, fixed 250 validation and 250 test
examples, and a fixed split seed of 101. Reverse-complement augmentation is
tested in the sweep via `use_reverse_complements: [false, true]`.

## Placeholder

`placeholder/utr_bassetvl/introns__placeholder__scratch__utr_bassetvl.yml.template`
is retained as a non-runnable naming/layout reference for future intron data
families.
