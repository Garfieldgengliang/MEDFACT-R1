#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 


export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"

QWEN_PATH="/root/autodl-tmp/QwenVL25_3B"
MIMIC_DATASET="/root/autodl-tmp/mimic_grpo_0306" 
OUTPUT_DIR="OUTPUT_DIR" 
RUN_NAME="MIMIC_QWEN25_3B_GRPO_20250306"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_lgl.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $MIMIC_DATASET \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 10000000 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4
