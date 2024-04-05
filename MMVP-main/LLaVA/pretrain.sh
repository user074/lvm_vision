#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed /home/jianing/Github/lvm_vision/LLaVA-main/scripts/zero2.json\
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /home/jianing/Github/lvm_vision/Data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /home/jianing/Github/lvm_vision/Data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_adopter_type cross_attention_8x \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
