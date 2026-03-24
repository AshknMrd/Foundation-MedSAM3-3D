#!/bin/bash

# ============================================================================
# SAM3 + LoRA Training for 3D NIfTI Data
# ============================================================================

CONFIG_DIR="configs/full_lora_config_3d_nifti.yaml"
SAM3_CHKPOINT="/mnt/scratch1/ashkanm/test_mine/workdir/checkpoints/sam3.pt"
INIT_LORA_WEIGHTS="/mnt/scratch1/ashkanm/test_mine/workdir/checkpoints/MedSAM3_v1/best_lora_weights.pt"


# OUTPUT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/zzzzzz"
# python train_sam3_lora_native_3d_nifti.py \
#     --config $CONFIG_DIR \
#     --sam3_chkdir $SAM3_CHKPOINT \
#     --init_lora_weights $INIT_LORA_WEIGHTS \
#     --output_dir $OUTPUT_DIR \
#     --num_epochs 2 \
#     --num_images 2 2 \
#     --all_slices \
#     --save_model_every_epoch

# ################################################################ #
# Train for 10 epochs and start from MedSAM3 LoRA weights
OUTPUT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/3D_train_infer/train/all_slices/E10_from_MedSAM3"
python train_sam3_lora_native_3d_nifti.py \
    --config $CONFIG_DIR \
    --sam3_chkdir $SAM3_CHKPOINT \
    --init_lora_weights $INIT_LORA_WEIGHTS \
    --output_dir $OUTPUT_DIR \
    --num_epochs 10 \
    --all_slices \
    --save_model_every_epoch
    
# Train for 10 epochs and start from Random LoRA weights
OUTPUT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/3D_train_infer/train/all_slices/E10_from_scratch"
python train_sam3_lora_native_3d_nifti.py \
    --config $CONFIG_DIR \
    --sam3_chkdir $SAM3_CHKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_epochs 10 \
    --all_slices \
    --save_model_every_epoch



