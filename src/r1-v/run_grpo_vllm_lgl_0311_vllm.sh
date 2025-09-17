#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 


export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"

QWEN_PATH="/root/autodl-tmp/Qwen2-5-VL-3B-Instruct"
# QWEN_PATH="/root/autodl-tmp/Qwen2-VL-2B-Instruct"
MIMIC_DATASET="/root/autodl-tmp/mimic_grpo_0307"
OUTPUT_DIR="OUTPUT_DIR" 
RUN_NAME="MIMIC_QWEN25_3B_GRPO_20250311"
# RUN_NAME="MIMIC_QWEN25_2B_GRPO_20250316"
DS_CONFIG="local_scripts/zero1_no_optimizer.json"
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2" torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_lgl_vllm.py \
    --output_dir $OUTPUT_DIR \
    --use_vllm true \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $MIMIC_DATASET \
    --max_prompt_length 4096  \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --num_generations 6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16  \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 501760 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_device "cuda:2" \
    --deepspeed local_scripts/zero3_2.json
    # --deepspeed local_scripts/zero1_no_optimizer.json
    
    # --deepspeed ${DS_CONFIG} \
