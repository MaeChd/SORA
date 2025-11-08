#!/bin/bash

export WORLD_SIZE=8
export SCRATCH=/home/exp_math
export TOKENIZERS_PARALLELISM=false

NUM_PROCESSES=${WORLD_SIZE}
ADAPTER_NAME=$1 
LEARNING_RATE=$2 
REGULARIZATION_LAMBDA=$3 
NUM_EPOCHS=$4 
RANK=$5 
RNAK_P=$6
LORA_ALPHA=$7 
DATASET_DIR=$8 
export ASCEND_RT_VISIBLE_DEVICES=$9

PORT=2025
SEED=0
ACTIVATE_PROFILING=False
WANDB_PROJECT=experiments_math
WANDB_MODE=disabled
PARAMETERIZE_S=identity
GRADIENT_TYPE=landing
INIT_STRATEGY=default
DTYPE=fp16

export WANDB_MODE=disabled
BASE_MODEL=/home/pretrained_weights/gemma-7b/
LR_SCHEDULER_TYPE=cosine
CUTOFF_LEN=512
CURRENT_DATE=$(date +"%Y%m%dT%H%M%S%3N")
echo $CURRENT_DATE
echo $ADAPTER_NAME
RUN_NAME=SORA_${CURRENT_DATE}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_DIR=$SCRATCH/${WANDB_PROJECT}/${RUN_NAME}
mkdir -p $RUN_DIR
WANDB_RUN_NAME=${RUN_NAME}

python -m torch.distributed.run --nproc_per_node=${NUM_PROCESSES} --master-port=${PORT} finetune_math.py \
	--base_model ${BASE_MODEL} \
	--data_path ${DATASET_DIR} \
	--output_dir $RUN_DIR \
    --adapter_name ${ADAPTER_NAME} \
	--activate_profiling ${ACTIVATE_PROFILING} \
	--batch_size 128 \
	--dtype ${DTYPE} \
	--micro_batch_size 1 \
	--num_epochs ${NUM_EPOCHS} \
	--learning_rate $LEARNING_RATE \
    --regularization_lambda $REGULARIZATION_LAMBDA \
	--lora_r ${RANK} \
	--lora_rp ${RANK_P} \
	--lora_alpha ${LORA_ALPHA} \
	--cutoff_len ${CUTOFF_LEN} \
	--lr_scheduler_type ${LR_SCHEDULER_TYPE} \
	--val_set_size 0.1 \
	--init_lora_weights ${INIT_STRATEGY} \
	--lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
	--adapter_name $ADAPTER_NAME \
	--parameterize_S ${PARAMETERIZE_S} \
	--gradient_type ${GRADIENT_TYPE} \
    --wandb_project ${WANDB_PROJECT} \
    --seed ${SEED} \
    --wandb_run_name ${WANDB_RUN_NAME} 2>&1 | tee "${RUN_DIR}/train.log"