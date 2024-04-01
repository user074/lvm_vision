#!/bin/bash
deepspeed  --include localhost:0,1,2,3,4,5,6,7\
    llava/train/train_mem.py \
    --deepspeed PATH_TO_DEEPSPEED\
    --model_name_or_path PATH_TO_VICUNAMODEL \
    --version v1 \
    --data_path PATH_TO_DATA \
    --image_folder PATH_TO_IMAGE_IN_DATA \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter PATH_TO_MM_ADAPTER \
    --pretrain_dino_mm_mlp_adapter PATH_TO_DINO_ADAPTER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir PATH_TO_OUTPUT_DIRECTORY \
    --num_train_epochs 1 \
    --per_device_train_batch_size 11 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
    --report_to wandb
#   --pretrain_dino_mm_mlp_adapter PATH_TO_DINO_ADAPTER \
#   --pretrain_fusion_adapter PATH_TO_FUSION_CrossAttention_ADAPTER \
