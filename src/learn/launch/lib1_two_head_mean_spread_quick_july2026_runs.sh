#!/usr/bin/env bash
set -euo pipefail

RUN_PART="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LEARN_ROOT="${REPO_ROOT}/src/learn"
PYTHON_BIN="${PYTHON:-python}"
RUN_TAG="lib1_two_head_mean_spread_quick_july2026"
LOGGER_PROJECT="${LOGGER_PROJECT:-${RUN_TAG}}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${LEARN_ROOT}"

COMMON_ARGS=(
  --data_module Lib1MeanSpreadDataModule
  --graph_module CNNWeightedRegressionTraining
  --sep tab
  --target_column mean_expr
  --target_columns mean_expr log_barcode_var
  --barcode_column n_barcodes
  --normalize true
  --test_min_barcodes 8
  --train_min_barcodes 1
  --train_size_frac 1.0
  --train_sampling_mode random
  --barcode_weighting true
  --barcode_weight_cap 8.0
  --barcode_weight_min 0.1
  --n_outputs 2
  --output_names mean_expr log_barcode_var
  --log_per_output_metric_details true
  --log_legacy_metric_aliases false
  --loss_criterion MSELoss
  --reduction mean
  --weighted_loss_reduction mean
  --logger_type wandb
  --logger_project "${LOGGER_PROJECT}"
  --exact_run_name true
  --epoch_eval_splits train val test
  --checkpoint_monitor val_pearson
  --stopping_mode max
  --model_seed 1701
  --split_seed 101
  --use_reverse_complements false
  --num_workers "${NUM_WORKERS}"
  --accelerator gpu
  --devices "${DEVICES}"
  --precision 32
)

run_promoter() {
  local part="promoter"
  local config_id="promoter_cfg011"
  local project_part="${LOGGER_PROJECT}__${part}"
  BODA_TASK_FAMILY="promoter" \
  BODA_TARGET_FAMILY="bashor_in_house_lib1_dedup_mean_spread_promoter" \
  BODA_COMPARISON_GROUP="${RUN_TAG}" \
  BODA_LAUNCH_SCRIPT="src/learn/launch/lib1_two_head_mean_spread_quick_july2026_runs.sh" \
  BODA_LAUNCH_NOTES="two_head_mean_spread_quick; source_config=${config_id}; source_prior=lib1_outer_seed_selected_barcode_weighted_june2026" \
  "${PYTHON_BIN}" train_wandb_log.py \
    "${COMMON_ARGS[@]}" \
    --model_module PromoterBassetVL \
    --artifact_path "${LEARN_ROOT}/local_artifacts/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --best_checkpoint_dir "${LEARN_ROOT}/outputs/hpo_runs/by_project/${project_part}/best_checkpoint_model" \
    --default_root_dir "${LEARN_ROOT}/outputs/hpo_runs/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --run_name "${RUN_TAG}__${part}__${config_id}__seed101" \
    --datafile_path "${LEARN_ROOT}/derived_data/promoter/bashor_in_house/lib1_promoter_allvalid_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv" \
    --sequence_column Promoter \
    --padded_seq_len 51 \
    --padding_mode neutral \
    --neutral_pad_char N \
    --val_frac_within_hq 0.1295 \
    --test_frac_within_hq 0.1295 \
    --val_size_within_hq 250 \
    --test_size_within_hq 250 \
    --input_len 51 \
    --batch_size 64 \
    --max_epochs 220 \
    --min_epochs 20 \
    --stopping_patience 35 \
    --optimizer AdamW \
    --lr 0.0003407452234699 \
    --weight_decay 0.0014447417907665 \
    --amsgrad false \
    --beta1 0.8754674185821312 \
    --beta2 0.9955927497914906 \
    --scheduler None \
    --scheduler_interval step \
    --conv1_channels 113 \
    --conv1_kernel_size 5 \
    --conv2_channels 54 \
    --conv2_kernel_size 9 \
    --conv3_channels 77 \
    --conv3_kernel_size 7 \
    --adaptive_pool_output_size 12 \
    --n_linear_layers 2 \
    --linear_channels 50 \
    --linear_activation LeakyReLU \
    --linear_dropout_p 0.65 \
    --use_batch_norm true \
    --use_weight_norm false
}

run_utr5() {
  local part="utr5"
  local config_id="utr5_cfg007"
  local project_part="${LOGGER_PROJECT}__${part}"
  BODA_TASK_FAMILY="utr5" \
  BODA_TARGET_FAMILY="bashor_in_house_lib1_dedup_mean_spread_utr5" \
  BODA_COMPARISON_GROUP="${RUN_TAG}" \
  BODA_LAUNCH_SCRIPT="src/learn/launch/lib1_two_head_mean_spread_quick_july2026_runs.sh" \
  BODA_LAUNCH_NOTES="two_head_mean_spread_quick; source_config=${config_id}; source_prior=lib1_outer_seed_selected_barcode_weighted_june2026" \
  "${PYTHON_BIN}" train_wandb_log.py \
    "${COMMON_ARGS[@]}" \
    --model_module ResNet1DRegressor \
    --artifact_path "${LEARN_ROOT}/local_artifacts/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --best_checkpoint_dir "${LEARN_ROOT}/outputs/hpo_runs/by_project/${project_part}/best_checkpoint_model" \
    --default_root_dir "${LEARN_ROOT}/outputs/hpo_runs/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --run_name "${RUN_TAG}__${part}__${config_id}__seed101" \
    --datafile_path "${LEARN_ROOT}/derived_data/utr5/bashor_in_house/lib1_fiveprime_modal50_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv" \
    --sequence_column FivePrime \
    --padded_seq_len 50 \
    --padding_mode none \
    --val_frac_within_hq 0.2 \
    --test_frac_within_hq 0.2 \
    --val_size_within_hq 250 \
    --test_size_within_hq 250 \
    --input_len 50 \
    --batch_size 128 \
    --max_epochs 220 \
    --min_epochs 20 \
    --stopping_patience 35 \
    --optimizer AdamW \
    --lr 0.00005357342404841717 \
    --weight_decay 0.000004523112179868298 \
    --amsgrad false \
    --beta1 0.8779363870477797 \
    --beta2 0.9903878429974556 \
    --scheduler CosineAnnealingWarmRestarts \
    --scheduler_interval step \
    --T_0 500 \
    --T_mult 1 \
    --eta_min 0 \
    --stem_channels 80 \
    --stem_kernel_size 5 \
    --block_kernel_size 3 \
    --dropout_p 0.395115230207692 \
    --head_hidden_channels 211 \
    --use_batch_norm false
}

run_intron() {
  local part="intron"
  local config_id="intron_cfg011"
  local project_part="${LOGGER_PROJECT}__${part}"
  BODA_TASK_FAMILY="intron" \
  BODA_TARGET_FAMILY="bashor_in_house_lib1_dedup_mean_spread_intron" \
  BODA_COMPARISON_GROUP="${RUN_TAG}" \
  BODA_LAUNCH_SCRIPT="src/learn/launch/lib1_two_head_mean_spread_quick_july2026_runs.sh" \
  BODA_LAUNCH_NOTES="two_head_mean_spread_quick; source_config=${config_id}; source_prior=lib1_outer_seed_selected_barcode_weighted_june2026" \
  "${PYTHON_BIN}" train_wandb_log.py \
    "${COMMON_ARGS[@]}" \
    --model_module ResNet1DRegressor \
    --artifact_path "${LEARN_ROOT}/local_artifacts/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --best_checkpoint_dir "${LEARN_ROOT}/outputs/hpo_runs/by_project/${project_part}/best_checkpoint_model" \
    --default_root_dir "${LEARN_ROOT}/outputs/hpo_runs/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --run_name "${RUN_TAG}__${part}__${config_id}__seed101" \
    --datafile_path "${LEARN_ROOT}/derived_data/introns/bashor_in_house/lib1_intron_modal80_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv" \
    --sequence_column Intron \
    --padded_seq_len 80 \
    --padding_mode none \
    --val_frac_within_hq 0.2 \
    --test_frac_within_hq 0.2 \
    --val_size_within_hq 250 \
    --test_size_within_hq 250 \
    --input_len 80 \
    --batch_size 256 \
    --max_epochs 180 \
    --min_epochs 20 \
    --stopping_patience 35 \
    --optimizer AdamW \
    --lr 0.000032709628322683204 \
    --weight_decay 0.0002483298893004 \
    --amsgrad true \
    --beta1 0.9393535025005216 \
    --beta2 0.9897327681659188 \
    --scheduler CosineAnnealingWarmRestarts \
    --scheduler_interval step \
    --T_0 2000 \
    --T_mult 1 \
    --eta_min 0 \
    --stem_channels 87 \
    --stem_kernel_size 7 \
    --block_kernel_size 5 \
    --dropout_p 0.3202972616738003 \
    --head_hidden_channels 55 \
    --use_batch_norm true
}

run_utr3() {
  local part="utr3"
  local config_id="utr3_cfg001"
  local project_part="${LOGGER_PROJECT}__${part}"
  BODA_TASK_FAMILY="utr3" \
  BODA_TARGET_FAMILY="bashor_in_house_lib1_dedup_mean_spread_utr3" \
  BODA_COMPARISON_GROUP="${RUN_TAG}" \
  BODA_LAUNCH_SCRIPT="src/learn/launch/lib1_two_head_mean_spread_quick_july2026_runs.sh" \
  BODA_LAUNCH_NOTES="two_head_mean_spread_quick; source_config=${config_id}; source_prior=lib1_outer_seed_selected_barcode_weighted_june2026" \
  "${PYTHON_BIN}" train_wandb_log.py \
    "${COMMON_ARGS[@]}" \
    --model_module ResNet1DRegressor \
    --artifact_path "${LEARN_ROOT}/local_artifacts/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --best_checkpoint_dir "${LEARN_ROOT}/outputs/hpo_runs/by_project/${project_part}/best_checkpoint_model" \
    --default_root_dir "${LEARN_ROOT}/outputs/hpo_runs/${RUN_TAG}/${part}/${config_id}/split_seed_101" \
    --run_name "${RUN_TAG}__${part}__${config_id}__seed101" \
    --datafile_path "${LEARN_ROOT}/derived_data/utr3/bashor_in_house/lib1_threeprime_modal100_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv" \
    --sequence_column ThreePrime \
    --padded_seq_len 100 \
    --padding_mode none \
    --val_frac_within_hq 0.25 \
    --test_frac_within_hq 0.25 \
    --input_len 100 \
    --batch_size 128 \
    --max_epochs 180 \
    --min_epochs 20 \
    --stopping_patience 30 \
    --optimizer AdamW \
    --lr 0.0002788209466101 \
    --weight_decay 0.000014642636720186382 \
    --amsgrad true \
    --beta1 0.9371043499777512 \
    --beta2 0.9933313982195612 \
    --scheduler CosineAnnealingWarmRestarts \
    --scheduler_interval step \
    --T_0 1000 \
    --T_mult 1 \
    --eta_min 0 \
    --stem_channels 127 \
    --stem_kernel_size 5 \
    --block_kernel_size 3 \
    --dropout_p 0.395251321300409 \
    --head_hidden_channels 253 \
    --use_batch_norm false
}

case "${RUN_PART}" in
  promoter)
    run_promoter
    ;;
  utr5|fiveprime|5utr)
    run_utr5
    ;;
  intron|introns)
    run_intron
    ;;
  utr3|threeprime|3utr)
    run_utr3
    ;;
  all)
    run_promoter
    run_utr5
    run_intron
    run_utr3
    ;;
  *)
    echo "Usage: $0 [all|promoter|utr5|intron|utr3]" >&2
    exit 2
    ;;
esac
