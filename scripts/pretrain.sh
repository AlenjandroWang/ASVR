#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1


# torchrun --nnodes=1 --nproc_per_node=2 \
#     train_mem.py \
#     --deepspeed ./scripts/zero2.json \

deepspeed ./train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --version plain \
    --data_path ./data/blip_laion_cc_sbu_558k.json \
    --image_folder ./data/LLaVA_pretrain \
    --vision_tower google/siglip-so400m-patch14-384 \
    --vision_tokenizer ./model_zoo/vision_tower_RVQ8_False_1152_epoch_6 \
    --vision_tokenizer_weight ./model_zoo/epoch_6.pt\
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --use_s2 False \
    --output_dir ./checkpoints/asvr-vicuna-1.5-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
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
    --report_to none | tee 2>&1 ./log_pretrain.txt \
