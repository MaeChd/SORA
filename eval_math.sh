#!/bin/bash

BASE_MODEL="/home/pretrained_weights/gemma-7b"
MODEL_NAME="Gemma-7B"
NPROC=8

declare -A ADAPTERS=(
    ["SoRA"]="16:PATH_TO_SORA_CHECKPOINT"
    ["LoRA"]="16:PATH_TO_LORA_CHECKPOINT"
    ["DoRA"]="16:PATH_TO_DORA_CHECKPOINT"
    ["OFT"]="4:PATH_TO_OFT_CHECKPOINT"
    ["LoRAPlus"]="16:PATH_TO_LORAPLUS_CHECKPOINT"
)

DATASETS=("gsm8k" "SVAMP" "MultiArith" "AQuA")

# Test all adapters and datasets if these arrays are empty
SELECTED_ADAPTERS=()
SELECTED_DATASETS=()

# get learning rate info from path
extract_lr_info() {
    local path="$1"
    if [[ $path =~ ([0-9]+\.?[0-9]*e-[0-9]+_[0-9]+\.?[0-9]*e-[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown_lr"
    fi
}

# get checkpoint number from path
extract_checkpoint_num() {
    local path="$1"
    if [[ $path =~ checkpoint-([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown_ckpt"
    fi
}


if [ ${#SELECTED_ADAPTERS[@]} -eq 0 ]; then
    TEST_ADAPTERS=("${!ADAPTERS[@]}")
else
    TEST_ADAPTERS=("${SELECTED_ADAPTERS[@]}")
fi


if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    TEST_DATASETS=("${DATASETS[@]}")
else
    TEST_DATASETS=("${SELECTED_DATASETS[@]}")
fi


echo "=========================================="
echo "Starting evaluation with the following configuration:"
echo "Model: $MODEL_NAME"
echo "Adapter: ${TEST_ADAPTERS[*]}"
echo "Datasets: ${TEST_DATASETS[*]}"
echo "=========================================="

for adapter in "${TEST_ADAPTERS[@]}"; do
    if [ -z "${ADAPTERS[$adapter]}" ]; then
        echo "Warning: Adapter '$adapter' not found in the configuration. Skipping."
        continue
    fi
    

    IFS=':' read -r lora_r checkpoint_path <<< "${ADAPTERS[$adapter]}"
    

    lr_info=$(extract_lr_info "$checkpoint_path")
    ckpt_num=$(extract_checkpoint_num "$checkpoint_path")
    
    for dataset in "${TEST_DATASETS[@]}"; do
        log_file="eval_gemma_${dataset,,}_${adapter,,}_lr${lr_info}_r${lora_r}_ckpt${ckpt_num}.log"
        
        echo "Start task: $adapter on $dataset"
        echo "  - LoRA rank: $lora_r"
        echo "  - Checkpoint: $checkpoint_path"
        echo "  - Learning rate: $lr_info"
        echo "  - Checkpoint number: $ckpt_num"
        echo "  - Log file: $log_file"
        
        torchrun --nproc_per_node=$NPROC evaluate_gemma_math.py \
            --model "$MODEL_NAME" \
            --adapter "$adapter" \
            --base_model "$BASE_MODEL" \
            --lora_weights "$checkpoint_path" \
            --lora_r "$lora_r" \
            --dataset "$dataset" \
            > "$log_file" 2>&1

        pid=$!
        echo "  - PID: $pid"
        echo ""
        sleep 2
    done
done

echo "=========================================="
echo "all tasks submitted."
echo "use 'ps -ef | grep python' to check running jobs."
echo "use 'tail -f <log_file>' to check log files."
echo "=========================================="
