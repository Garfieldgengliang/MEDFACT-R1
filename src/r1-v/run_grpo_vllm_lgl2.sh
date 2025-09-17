#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 

# 参数配置（引用自原始命令行参数）
export OUTPUT_DIR="/root/autodl-tmp/OUTPUT_DIR"  # 输出目录需提前创建[1,3](@ref)
export QWEN_PATH="/root/autodl-tmp/QwenVL25_3B"  # 模型路径需确认存在[4,9](@ref)
export MIMIC_DATASET="/root/autodl-tmp/mimic_grpo_0306"  # 数据集路径需确认存在[2,7](@ref)
export RUN_NAME="MIMIC_QWEN25_3B_GRPO_20250306"  # 运行名称[5,6](@ref)

# 分布式训练配置（关键修改点）
CUDA_VISIBLE_DEVICES="2" torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12325" \
    src/open_r1/grpo_lgl.py \
    --output_dir "$OUTPUT_DIR" \  # 注意路径引号防止空格问题[3,8](@ref)
    --model_name_or_path "$QWEN_PATH" \
    --dataset_name "$MIMIC_DATASET" \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --deepspeed "/root/autodl-tmp/R1-V-main/R1-V-main/src/r1-v/local_scripts/zero3.json" \
    --attn_implementation flash_attention_2 \
    --max_pixels 8000000 \
    --max_steps 13125 \
    --run_name "$RUN_NAME" \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 4  # 与初始参数保持一致[1,6](@ref)