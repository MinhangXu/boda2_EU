#!/usr/bin/env bash

# Base directory for artifact output - update this to your preferred location
BASE_OUTPUT_DIR="/home/minhang/synBio_AL/boda2_EU/src/HPO_sup3_enhancer"

# Common parameters that don't change between models
DATA_MODULE="MPRA_DataModule"
DATAFILE_PATH="/home/minhang/synBio_AL/opt_EU_learn_n_design/CRE/Table_S2__MPRA_dataset.txt"
SEP="tab"
SEQUENCE_COLUMN="sequence"
ACTIVITY_COLUMNS="K562_log2FC HepG2_log2FC SKNSH_log2FC"
STDERR_COLUMNS="K562_lfcSE HepG2_lfcSE SKNSH_lfcSE"
VAL_CHRS="19 21 X"
TEST_CHRS="7 13"
USE_REVERSE_COMPLEMENTS="True"
NUM_WORKERS="8"
STD_MULTIPLE_CUT="6.0"
UP_CUTOFF_MOVE="3"

# Common model architecture parameters
INPUT_LEN="600"
PADDED_SEQ_LEN="600"
CONV1_CHANNELS="300"
CONV1_KERNEL_SIZE="19"
CONV2_CHANNELS="200"
CONV2_KERNEL_SIZE="11"
CONV3_CHANNELS="200"
CONV3_KERNEL_SIZE="7"
LINEAR_CHANNELS="1000"
LINEAR_ACTIVATION="ReLU"
N_OUTPUTS="3"
LOSS_CRITERION="L1KLmixed"
USE_BATCH_NORM="True"
USE_WEIGHT_NORM="False"
CRITERION_REDUCTION="mean"
PARENT_WEIGHTS="/home/minhang/synBio_AL/boda2_EU/src/my-model.epoch_5-step_19885.pkl"
GRAPH_MODULE="CNNTransferLearning"

# Common training parameters
OPTIMIZER="Adam"
SCHEDULER="CosineAnnealingWarmRestarts"
SCHEDULER_INTERVAL="step"
T_MULT="1"
ETA_MIN="0.0"
LAST_EPOCH="-1"
CHECKPOINT_MONITOR="epoch_end_r2"
STOPPING_MODE="max"
STOPPING_PATIENCE="30"
ACCELERATOR="gpu"
DEVICES="1"
MIN_EPOCHS="60"
MAX_EPOCHS="250"
PRECISION="16"
DEFAULT_ROOT_DIR="/tmp/output/artifacts"
ARTIFACT_PATH="/home/minhang/synBio_AL/boda2_EU/src/local_artifacts"

# logger parameters
LOGGER_TYPE="wandb"
LOGGER_PROJECT="replicate_HPO_enhancer"

# Choose GPU
GPU="0"  # Change this to your preferred GPU ID

# First model: 20240106_062017 (Best overall model)
MODEL_NAME="model_20240106_062017_BassetBranched"
MODEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "Training model based on timestamp model_20240106_062017_BassetBranched..."

# Model-specific parameters from the BO results
MODEL_MODULE="BassetBranched"
BATCH_SIZE="901"
DUPLICATION_CUTOFF="4.776607034319936"
N_LINEAR_LAYERS="2"
LINEAR_DROPOUT_P="0.05"
N_BRANCHED_LAYERS="4"
BRANCHED_CHANNELS="1024"
BRANCHED_ACTIVATION="ReLU6"
BRANCHED_DROPOUT_P="0.4899004908291405"
FROZEN_EPOCHS="33"
LR="0.0019349137200938"
WEIGHT_DECAY="0.0001996325108651"
AMSGRAD="True"
BETA="1.5108101458180505"
BETA1="0.9532963313781776"
BETA2="0.8006505529938938"
T_0="3512"
RUN_NAME="model_20240106_062017_BassetBranched"

CUDA_VISIBLE_DEVICES=$GPU python /home/minhang/synBio_AL/boda2_EU/src/train_wandb_log.py \
  --data_module="$DATA_MODULE" \
  --datafile_path="$DATAFILE_PATH" \
  --sep="$SEP" \
  --sequence_column="$SEQUENCE_COLUMN" \
  --activity_columns $ACTIVITY_COLUMNS \
  --stderr_columns $STDERR_COLUMNS \
  --batch_size="$BATCH_SIZE" \
  --duplication_cutoff="$DUPLICATION_CUTOFF" \
  --std_multiple_cut="$STD_MULTIPLE_CUT" \
  --up_cutoff_move="$UP_CUTOFF_MOVE" \
  --val_chrs $VAL_CHRS \
  --test_chrs $TEST_CHRS \
  --padded_seq_len="$PADDED_SEQ_LEN" \
  --use_reverse_complements="$USE_REVERSE_COMPLEMENTS" \
  --num_workers="$NUM_WORKERS" \
  --model_module="$MODEL_MODULE" \
  --input_len="$INPUT_LEN" \
  --conv1_channels="$CONV1_CHANNELS" \
  --conv1_kernel_size="$CONV1_KERNEL_SIZE" \
  --conv2_channels="$CONV2_CHANNELS" \
  --conv2_kernel_size="$CONV2_KERNEL_SIZE" \
  --conv3_channels="$CONV3_CHANNELS" \
  --conv3_kernel_size="$CONV3_KERNEL_SIZE" \
  --n_linear_layers="$N_LINEAR_LAYERS" \
  --linear_channels="$LINEAR_CHANNELS" \
  --linear_activation="$LINEAR_ACTIVATION" \
  --linear_dropout_p="$LINEAR_DROPOUT_P" \
  --n_branched_layers="$N_BRANCHED_LAYERS" \
  --branched_channels="$BRANCHED_CHANNELS" \
  --branched_activation="$BRANCHED_ACTIVATION" \
  --branched_dropout_p="$BRANCHED_DROPOUT_P" \
  --n_outputs="$N_OUTPUTS" \
  --loss_criterion="$LOSS_CRITERION" \
  --beta="$BETA" \
  --use_batch_norm="$USE_BATCH_NORM" \
  --use_weight_norm="$USE_WEIGHT_NORM" \
  --reduction="$CRITERION_REDUCTION" \
  --graph_module="$GRAPH_MODULE" \
  --parent_weights="$PARENT_WEIGHTS" \
  --frozen_epochs="$FROZEN_EPOCHS" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --weight_decay="$WEIGHT_DECAY" \
  --amsgrad="$AMSGRAD" \
  --beta1="$BETA1" \
  --beta2="$BETA2" \
  --scheduler="$SCHEDULER" \
  --scheduler_interval="$SCHEDULER_INTERVAL" \
  --T_0="$T_0" \
  --T_mult="$T_MULT" \
  --eta_min="$ETA_MIN" \
  --last_epoch="$LAST_EPOCH" \
  --checkpoint_monitor="$CHECKPOINT_MONITOR" \
  --stopping_mode="$STOPPING_MODE" \
  --stopping_patience="$STOPPING_PATIENCE" \
  --accelerator="$ACCELERATOR" \
  --devices="$DEVICES" \
  --min_epochs="$MIN_EPOCHS" \
  --max_epochs="$MAX_EPOCHS" \
  --precision="$PRECISION" \
  --default_root_dir="$MODEL_OUTPUT_DIR" \
  --artifact_path="$ARTIFACT_PATH/$MODEL_NAME" \
  --logger_type="$LOGGER_TYPE" \
  --logger_project="$LOGGER_PROJECT" \
  --run_name="$RUN_NAME"

echo "Model 1 (20240106_062017) training completed."

# Second model: 20240104_071417
MODEL_NAME="model_20240104_071417_BassetBranched"
MODEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "Training model based on timestamp model_20240104_071417_BassetBranched..."

# Model-specific parameters from the BO results
MODEL_MODULE="BassetBranched"
BATCH_SIZE="734"
DUPLICATION_CUTOFF="5.0"
N_LINEAR_LAYERS="3"
LINEAR_DROPOUT_P="0.0566670460912049"
N_BRANCHED_LAYERS="3"
BRANCHED_CHANNELS="1023"
BRANCHED_ACTIVATION="ELU"
BRANCHED_DROPOUT_P="0.4568292372759414"
FROZEN_EPOCHS="27"
LR="0.0018183393415252"
WEIGHT_DECAY="0.0002792765133892"
AMSGRAD="True"
BETA="2.664484816482072"
BETA1="0.9512078572191207"
BETA2="0.8000000000000002"
T_0="3950"
RUN_NAME="model_20240104_071417_BassetBranched"

CUDA_VISIBLE_DEVICES=$GPU python /home/minhang/synBio_AL/boda2_EU/src/train_wandb_log.py \
  --data_module="$DATA_MODULE" \
  --datafile_path="$DATAFILE_PATH" \
  --sep="$SEP" \
  --sequence_column="$SEQUENCE_COLUMN" \
  --activity_columns $ACTIVITY_COLUMNS \
  --stderr_columns $STDERR_COLUMNS \
  --batch_size="$BATCH_SIZE" \
  --duplication_cutoff="$DUPLICATION_CUTOFF" \
  --std_multiple_cut="$STD_MULTIPLE_CUT" \
  --up_cutoff_move="$UP_CUTOFF_MOVE" \
  --val_chrs $VAL_CHRS \
  --test_chrs $TEST_CHRS \
  --padded_seq_len="$PADDED_SEQ_LEN" \
  --use_reverse_complements="$USE_REVERSE_COMPLEMENTS" \
  --num_workers="$NUM_WORKERS" \
  --model_module="$MODEL_MODULE" \
  --input_len="$INPUT_LEN" \
  --conv1_channels="$CONV1_CHANNELS" \
  --conv1_kernel_size="$CONV1_KERNEL_SIZE" \
  --conv2_channels="$CONV2_CHANNELS" \
  --conv2_kernel_size="$CONV2_KERNEL_SIZE" \
  --conv3_channels="$CONV3_CHANNELS" \
  --conv3_kernel_size="$CONV3_KERNEL_SIZE" \
  --n_linear_layers="$N_LINEAR_LAYERS" \
  --linear_channels="$LINEAR_CHANNELS" \
  --linear_activation="$LINEAR_ACTIVATION" \
  --linear_dropout_p="$LINEAR_DROPOUT_P" \
  --n_branched_layers="$N_BRANCHED_LAYERS" \
  --branched_channels="$BRANCHED_CHANNELS" \
  --branched_activation="$BRANCHED_ACTIVATION" \
  --branched_dropout_p="$BRANCHED_DROPOUT_P" \
  --n_outputs="$N_OUTPUTS" \
  --loss_criterion="$LOSS_CRITERION" \
  --beta="$BETA" \
  --use_batch_norm="$USE_BATCH_NORM" \
  --use_weight_norm="$USE_WEIGHT_NORM" \
  --reduction="$CRITERION_REDUCTION" \
  --graph_module="$GRAPH_MODULE" \
  --parent_weights="$PARENT_WEIGHTS" \
  --frozen_epochs="$FROZEN_EPOCHS" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --weight_decay="$WEIGHT_DECAY" \
  --amsgrad="$AMSGRAD" \
  --beta1="$BETA1" \
  --beta2="$BETA2" \
  --scheduler="$SCHEDULER" \
  --scheduler_interval="$SCHEDULER_INTERVAL" \
  --T_0="$T_0" \
  --T_mult="$T_MULT" \
  --eta_min="$ETA_MIN" \
  --last_epoch="$LAST_EPOCH" \
  --checkpoint_monitor="$CHECKPOINT_MONITOR" \
  --stopping_mode="$STOPPING_MODE" \
  --stopping_patience="$STOPPING_PATIENCE" \
  --accelerator="$ACCELERATOR" \
  --devices="$DEVICES" \
  --min_epochs="$MIN_EPOCHS" \
  --max_epochs="$MAX_EPOCHS" \
  --precision="$PRECISION" \
  --default_root_dir="$MODEL_OUTPUT_DIR" \
  --artifact_path="$ARTIFACT_PATH/$MODEL_NAME" \
  --logger_type="$LOGGER_TYPE" \
  --logger_project="$LOGGER_PROJECT" \
  --run_name="$RUN_NAME"

echo "Model 2 (20240104_071417) training completed."

# Third model: 20240106_024527
MODEL_NAME="model_20240106_024527_BassetVL"
MODEL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "Training model based on timestamp model_20240106_024527_BassetVL..."

# Model-specific parameters from the BO results
MODEL_MODULE="BassetVL"
BATCH_SIZE="901"
DUPLICATION_CUTOFF="4.994840852431167"
N_LINEAR_LAYERS="3"
LINEAR_DROPOUT_P="0.05"
N_BRANCHED_LAYERS=""
BRANCHED_CHANNELS=""
BRANCHED_ACTIVATION=""
BRANCHED_DROPOUT_P=""
FROZEN_EPOCHS="46"
LR="0.002045814585304"
WEIGHT_DECAY="0.0002400734809518"
AMSGRAD="True"
BETA="0.2"
BETA1="0.8569764996514223"
BETA2="0.9232373053730729"
T_0="2832"
RUN_NAME="model_20240106_024527_BassetVL"

CUDA_VISIBLE_DEVICES=$GPU python /home/minhang/synBio_AL/boda2_EU/src/train_wandb_log.py \
  --data_module="$DATA_MODULE" \
  --datafile_path="$DATAFILE_PATH" \
  --sep="$SEP" \
  --sequence_column="$SEQUENCE_COLUMN" \
  --activity_columns $ACTIVITY_COLUMNS \
  --stderr_columns $STDERR_COLUMNS \
  --batch_size="$BATCH_SIZE" \
  --duplication_cutoff="$DUPLICATION_CUTOFF" \
  --std_multiple_cut="$STD_MULTIPLE_CUT" \
  --up_cutoff_move="$UP_CUTOFF_MOVE" \
  --val_chrs $VAL_CHRS \
  --test_chrs $TEST_CHRS \
  --padded_seq_len="$PADDED_SEQ_LEN" \
  --use_reverse_complements="$USE_REVERSE_COMPLEMENTS" \
  --num_workers="$NUM_WORKERS" \
  --model_module="$MODEL_MODULE" \
  --input_len="$INPUT_LEN" \
  --conv1_channels="$CONV1_CHANNELS" \
  --conv1_kernel_size="$CONV1_KERNEL_SIZE" \
  --conv2_channels="$CONV2_CHANNELS" \
  --conv2_kernel_size="$CONV2_KERNEL_SIZE" \
  --conv3_channels="$CONV3_CHANNELS" \
  --conv3_kernel_size="$CONV3_KERNEL_SIZE" \
  --n_linear_layers="$N_LINEAR_LAYERS" \
  --linear_channels="$LINEAR_CHANNELS" \
  --linear_activation="$LINEAR_ACTIVATION" \
  --linear_dropout_p="$LINEAR_DROPOUT_P" \
  --n_outputs="$N_OUTPUTS" \
  --loss_criterion="$LOSS_CRITERION" \
  --beta="$BETA" \
  --use_batch_norm="$USE_BATCH_NORM" \
  --use_weight_norm="$USE_WEIGHT_NORM" \
  --reduction="$CRITERION_REDUCTION" \
  --graph_module="$GRAPH_MODULE" \
  --parent_weights="$PARENT_WEIGHTS" \
  --frozen_epochs="$FROZEN_EPOCHS" \
  --optimizer="$OPTIMIZER" \
  --lr="$LR" \
  --weight_decay="$WEIGHT_DECAY" \
  --amsgrad="$AMSGRAD" \
  --beta1="$BETA1" \
  --beta2="$BETA2" \
  --scheduler="$SCHEDULER" \
  --scheduler_interval="$SCHEDULER_INTERVAL" \
  --T_0="$T_0" \
  --T_mult="$T_MULT" \
  --eta_min="$ETA_MIN" \
  --last_epoch="$LAST_EPOCH" \
  --checkpoint_monitor="$CHECKPOINT_MONITOR" \
  --stopping_mode="$STOPPING_MODE" \
  --stopping_patience="$STOPPING_PATIENCE" \
  --accelerator="$ACCELERATOR" \
  --devices="$DEVICES" \
  --min_epochs="$MIN_EPOCHS" \
  --max_epochs="$MAX_EPOCHS" \
  --precision="$PRECISION" \
  --default_root_dir="$MODEL_OUTPUT_DIR" \
  --artifact_path="$ARTIFACT_PATH/$MODEL_NAME" \
  --logger_type="$LOGGER_TYPE" \
  --logger_project="$LOGGER_PROJECT" \
  --run_name="$RUN_NAME"

echo "Model 3 (20240106_024527) training completed."
echo "All models training finished!"
