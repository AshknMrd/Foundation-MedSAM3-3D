#!/bin/bash

## MedSAM3+LoRA Prediction:

CONFIG_DIR="configs/full_lora_config_3d_nifti.yaml"
SAM3_CHKPOINT="/mnt/scratch1/ashkanm/test_mine/workdir/checkpoints/sam3.pt"
INPUT_DIR="/mnt/scratch1/ashkanm/move2scratch/Dataset_new_preprocessed/Testset_AMOS/imagesTs"
LABEL_DIR="/mnt/scratch1/ashkanm/move2scratch/Dataset_new_preprocessed/Testset_AMOS/labelsTs_spleen"

# NUM_IMGS=1

OUTPUT_IMG="/mnt/scratch1/ashkanm/test_mine/z_lora_last_all_slices_amos/E10_from_MedSAM3"
for EPOCH in {1..10}; do
    LORA_CHKPOINT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/3D_train_infer/train/all_slices/E10_from_MedSAM3/epoch_${EPOCH}_lora_weights.pt"
    python infer_sam3_plus_lora_3d_nifti.py --config $CONFIG_DIR --sam3_chk $SAM3_CHKPOINT --weights $LORA_CHKPOINT_DIR --images_dir $INPUT_DIR --labels_dir $LABEL_DIR --prompt "spleen" --output_dir $OUTPUT_IMG --all_slices --no_save_nifti # --num_images $NUM_IMGS
    mv ${OUTPUT_IMG}/inference_results.json ${OUTPUT_IMG}/inference_results${EPOCH}.json
done

OUTPUT_IMG="/mnt/scratch1/ashkanm/test_mine/z_lora_last_all_slices_amos/E10_from_scratch"
for EPOCH in {1..10}; do
    LORA_CHKPOINT_DIR="/mnt/scratch1/ashkanm/test_mine/workdir/output/sam3_plus_lora/3D_train_infer/train/all_slices/E10_from_scratch/epoch_${EPOCH}_lora_weights.pt"
    python infer_sam3_plus_lora_3d_nifti.py --config $CONFIG_DIR --sam3_chk $SAM3_CHKPOINT --weights $LORA_CHKPOINT_DIR --images_dir $INPUT_DIR --labels_dir $LABEL_DIR --prompt "spleen" --output_dir $OUTPUT_IMG --all_slices --no_save_nifti # --num_images $NUM_IMGS 
    mv ${OUTPUT_IMG}/inference_results.json ${OUTPUT_IMG}/inference_results${EPOCH}.json
done

## nnUNet Prediction:
eval "$(conda shell.bash hook)"
conda activate /mnt/scratch1/ashkanm/nnunet_env

export nnUNet_raw="/mnt/scratch1/ashkanm/test_mine/nnunet_workdir/nnUNet_raw"
export nnUNet_preprocessed="/mnt/scratch1/ashkanm/test_mine/nnunet_workdir/nnUNet_preprocessed"
export nnUNet_results="/mnt/scratch1/ashkanm/test_mine/nnunet_workdir/nnUNet_results"

DATASET_ID=109
DATASET_NAME="Dataset109_Spleen"
CONFIGURATION="3d_fullres"
TRAINER_NAME="nnUNetTrainerDiceLoss_300epochs"

# ## Prediction and Evaluate ######################################################################
INPUT_IMGS="/mnt/scratch1/ashkanm/test_mine/nnunet_workdir/nnUNet_raw/Testset_AMOS/imagesTs"
GT_FOLDER="/mnt/scratch1/ashkanm/test_mine/nnunet_workdir/nnUNet_raw/Testset_AMOS/labelsTs_spleen"
OUTPUT_DIR="/mnt/scratch1/ashkanm/test_mine/z_lora_last_all_slices_amos"
PRED_FOLDER="${OUTPUT_DIR}/predictions"
DJFILE="${PRED_FOLDER}/dataset.json"
PFILE="${PRED_FOLDER}/plans.json"

if [ ! -d "$PRED_FOLDER" ]; then
    echo "Output is created!"
    mkdir -p "$PRED_FOLDER"
fi

for EPOCH in {25,51,77,103,129,155,181,207,233,259}; do
    CHKPOINT="checkpoint_e${EPOCH}.pth"
    nnUNetv2_predict -i $INPUT_IMGS -o $PRED_FOLDER -d $DATASET_ID -tr $TRAINER_NAME -c $CONFIGURATION -chk $CHKPOINT #--save_probabilities
    nnUNetv2_evaluate_folder $GT_FOLDER $PRED_FOLDER -djfile $DJFILE -pfile $PFILE
    mv ${PRED_FOLDER}/summary.json ${OUTPUT_DIR}/summary_E${EPOCH}.json
    rm -rf $PRED_FOLDER
done