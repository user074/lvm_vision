#!/bin/bash
deepspeed llava/train/train_mem.py \
    --deepspeed /home/jianing/Github/lvm_vision/LLaVA-main/scripts/zero3.json\
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --version v1 \
    --data_path /home/jianing/Github/lvm_vision/Data/LLaVA-finetune/llava_v1_5_mix665k.json \
    --image_folder /home/jianing/Github/lvm_vision/Data/LLaVA-finetune \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/jianing/Github/lvm_vision/MMVP-main/LLaVA/checkpoints/tinyLlama-1.1B-pretrain/mm_projector.bin \
    --pretrain_dino_mm_mlp_adapter /home/jianing/Github/lvm_vision/MMVP-main/LLaVA/checkpoints/tinyLlama-1.1B-pretrain/dino_mm_projector.bin \
    --pretrain_fusion_adapter /home/jianing/Github/lvm_vision/MMVP-main/LLaVA/checkpoints/tinyLlama-1.1B-pretrain/fusion_adapter.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_adopter_type cross_attention_8x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/tinyLlama-1.1B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb
#   --pretrain_dino_mm_mlp_adapter PATH_TO_DINO_ADAPTER \
#   --pretrain_fusion_adapter PATH_TO_FUSION_CrossAttention_ADAPTER \
