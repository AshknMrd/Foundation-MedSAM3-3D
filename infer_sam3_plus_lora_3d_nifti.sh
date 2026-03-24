#!/bin/bash

# ============================================================================
# SAM3 + LoRA Inference for 2D JPG Data
# ============================================================================
# CONFIG_DIR="configs/full_lora_config_3d_nifti.yaml"
# SAM3_CHKPOINT="/mnt/scratch1/ashkanm/test_mine/workdir/checkpoints/sam3.pt"
# LORA_CHKPOINT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/simple_sam3_lora/best_lora_weights.pt"
# INPUT_IMG="/mnt/scratch1/ashkanm/test_mine/workdir/output/infer/image_slice.jpg"
# LABEL_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/infer/label_slice.jpg"
# OUTPUT_IMG="/mnt/scratch1/ashkanm/test_mine/workdir/output/infer/output_mylora.png"

# # Inference SAM3
# python infer_sam3_plus_lora_mine.py \
#     --sam3_chk $SAM3_CHKPOINT \
#     --weights $LORA_CHKPOINT_DIR \
#     --config $CONFIG_DIR \
#     --image $INPUT_IMG \
#     --label $LABEL_DIR \
#     --prompt "spleen" \
#     --output $OUTPUT_IMG

# ============================================================================
# SAM3 + LoRA Inference for 3D NIfTI Data
# ============================================================================
CONFIG_DIR="configs/full_lora_config_3d_nifti.yaml"
SAM3_CHKPOINT="/mnt/scratch1/ashkanm/test_mine/workdir/checkpoints/sam3.pt"
LORA_CHKPOINT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/2D_train_infer/train/best_lora_weights.pt"
INPUT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/dataset/Task09_Spleen/imagesTs"
LABEL_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/dataset/Task09_Spleen/labelsTs"
OUTPUT_IMG="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/3D_train_infer/infer"
TEST_NUM=2

# Inference SAM3
python infer_sam3_plus_lora_3d_nifti.py \
    --config $CONFIG_DIR \
    --sam3_chk $SAM3_CHKPOINT \
    --weights $LORA_CHKPOINT_DIR \
    --images_dir $INPUT_DIR \
    --labels_dir $LABEL_DIR \
    --prompt "spleen" \
    --num_images $TEST_NUM \
    --visualize \
    --all_slices \
    --no_save_nifti \
    --output_dir $OUTPUT_IMG 

