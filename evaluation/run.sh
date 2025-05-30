#!/usr/bin/bash
TASKS="Cryosphere"
MODEL="qwen2_5_vl"
PRETRAINED_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ARGS="pretrained=${PRETRAINED_MODEL},use_flash_attention_2=True"
LOG_SUFFIX="${MODEL}_${TASKS}"

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model "qwen2_5_vl" \
    --model_args ${MODEL_ARGS}  \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${LOG_SUFFIX} \
    --output_path ./logs/