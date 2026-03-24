#!/usr/bin/env python3
"""
SAM3 + LoRA 3D NIfTI Folder Inference Script

Runs inference on a folder of NIfTI 3D volumes, slice by slice,
and computes per-volume and overall Dice scores.

Based on the 2D slice inference (infer_sam3_plus_lora_2D_slice.py) and
the NIfTI training pipeline (train_sam3_lora_native_3d_nifti_folder.py).

Usage:
    # Basic inference on a NIfTI folder
    python3 infer_sam3_plus_lora_3d_nifti_folder.py \
        --config configs/full_lora_config_3d_nifti.yaml \
        --sam3_chk /workspace/sam3.pt \
        --weights /workspace/output/best_lora_weights.pt \
        --images_dir /data/imagesTs \
        --labels_dir /data/labelsTs \
        --prompt "lesion" \
        --output_dir /workspace/inference_output

    # With custom threshold, axis, and num_images limit
    python3 infer_sam3_plus_lora_3d_nifti_folder.py \
        --config configs/full_lora_config_3d_nifti.yaml \
        --sam3_chk /workspace/sam3.pt \
        --weights /workspace/output/best_lora_weights.pt \
        --images_dir /data/imagesTs \
        --labels_dir /data/labelsTs \
        --prompt "tumor" \
        --axis axial \
        --threshold 0.5 \
        --nms-iou 0.5 \
        --num_images 5 \
        --output_dir /workspace/inference_output
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms
from tqdm import tqdm

import nibabel as nib
from scipy import ndimage

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


# ============================================================================
# NIfTI Volume Utilities (same style as train script)
# ============================================================================

def normalize_slice(slice_2d):
    """Normalize 2D slice to [0, 255] uint8 (same as NIfTIDataset._normalize_slice)"""
    slice_2d = slice_2d.astype(np.float32)
    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0)

    # Percentile normalization
    p_low, p_high = np.percentile(slice_2d, [1, 99])
    if p_high > p_low:
        slice_2d = np.clip(slice_2d, p_low, p_high)
        slice_2d = (slice_2d - p_low) / (p_high - p_low) * 255.0
    else:
        slice_2d = np.zeros_like(slice_2d)

    return slice_2d.astype(np.uint8)


def extract_slice(volume, slice_idx, axis):
    """Extract 2D slice from 3D volume (same as NIfTIDataset._extract_slice)"""
    if axis == 'axial':
        return volume[:, :, slice_idx]
    elif axis == 'sagittal':
        return volume[slice_idx, :, :]
    else:  # coronal
        return volume[:, slice_idx, :]


def get_num_slices(volume, axis):
    """Get number of slices along the given axis"""
    axis_map = {'axial': 2, 'sagittal': 0, 'coronal': 1}
    return volume.shape[axis_map[axis]]


def compute_dice_score(pred_mask, gt_mask):
    """
    Compute Dice score between prediction and ground truth binary masks.

    Args:
        pred_mask: Binary prediction mask (np.ndarray)
        gt_mask: Binary ground truth mask (np.ndarray)

    Returns:
        Dice score (float)
    """
    pred_binary = pred_mask.astype(bool)
    gt_binary = gt_mask.astype(bool)

    intersection = np.sum(pred_binary & gt_binary)
    sum_masks = np.sum(pred_binary) + np.sum(gt_binary)

    if sum_masks == 0:
        # Both empty => perfect match
        return 1.0

    dice = (2.0 * intersection) / (sum_masks + 1e-8)
    return float(dice)


# ============================================================================
# SAM3 + LoRA NIfTI Inference
# ============================================================================

class SAM3LoRANIfTIInference:
    """SAM3 model with LoRA for 3D NIfTI volume inference."""

    def __init__(
        self,
        config_path: str,
        sam3_checkpoint: str,
        weights_path: str,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize SAM3 with LoRA for NIfTI inference.

        Args:
            config_path: Path to training config YAML
            sam3_checkpoint: Path to SAM3 base model checkpoint
            weights_path: Path to LoRA weights
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to run on (default: "cuda")
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sam3_checkpoint = sam3_checkpoint
        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"🔧 Initializing SAM3 + LoRA for NIfTI inference...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        # Build base model
        print("\n📦 Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=self.sam3_checkpoint,
            load_from_HF=False,
            eval_mode=True
        )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        if weights_path is not None:
            print(f"💾 Loading LoRA weights from {weights_path}...")
            load_lora_weights(self.model, weights_path)
        else:
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern, same as 2D inference)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        print("✅ SAM3 + LoRA ready for NIfTI inference!\n")

    def create_datapoint(self, pil_image: PILImage.Image, text_prompt: str, image_id: int = 0) -> Datapoint:
        """
        Create a SAM3 datapoint from a PIL image and text prompt.
        (Same pattern as infer_sam3_plus_lora_2D_slice.py)

        Args:
            pil_image: PIL Image (RGB)
            text_prompt: Text query string
            image_id: Image identifier

        Returns:
            Datapoint with image and query
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create query
        query = FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=image_id,
                original_image_id=image_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )

        return Datapoint(find_queries=[query], images=[sam_image])

    @torch.no_grad()
    def predict_slice(self, pil_image: PILImage.Image, text_prompt: str, image_id: int = 0) -> Optional[np.ndarray]:
        """
        Run inference on a single 2D slice.
        (Same logic as SAM3LoRAInference.predict but returns a single binary mask)

        Args:
            pil_image: PIL Image (RGB)
            text_prompt: Text query
            image_id: Image identifier

        Returns:
            Binary mask at original resolution (np.ndarray, bool) or None if no detections
        """
        orig_w, orig_h = pil_image.size

        # Create datapoint
        datapoint = self.create_datapoint(pil_image, text_prompt, image_id)

        # Apply transforms
        datapoint = self.transform(datapoint)

        # Collate into batch
        batch = collate_fn_api([datapoint], dict_key="input")["input"]

        # Move to device
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(batch)

        # Post-processing (same as 2D inference script)
        last_output = outputs[-1]
        pred_logits = last_output['pred_logits']   # [batch, num_queries, num_classes]
        pred_boxes = last_output['pred_boxes']      # [batch, num_queries, 4]
        pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

        # Get probabilities
        out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

        # Get scores for this query
        scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

        # Filter by threshold
        keep = scores > self.detection_threshold
        num_keep = keep.sum().item()

        if num_keep == 0 or pred_masks is None:
            return None

        # Get boxes and convert from cxcywh to xyxy
        boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
        kept_scores = scores[keep]
        cx, cy, w, h = boxes_cxcywh.unbind(-1)

        # Convert to xyxy and scale to original image size
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Apply NMS to remove overlapping boxes
        keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
        kept_scores = kept_scores[keep_nms]

        # Get masks and resize to original size
        masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

        # Resize masks to original image size
        masks_resized = F.interpolate(
            masks_small.unsqueeze(0).float(),
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) > 0.5  # [num_keep_nms, orig_h, orig_w]

        # Combine all masks into a single binary mask (union)
        combined_mask = masks_resized.any(dim=0).cpu().numpy()  # [orig_h, orig_w]

        return combined_mask

    def predict_volume(
        self,
        img_file: Path,
        label_file: Optional[Path],
        text_prompt: str,
        axis: str = 'axial',
        all_slices: bool = False
    ) -> dict:
        """
        Run inference on an entire NIfTI volume, slice by slice.

        Args:
            img_file: Path to NIfTI image file
            label_file: Path to NIfTI label file (optional, for Dice computation)
            text_prompt: Text prompt for segmentation
            axis: Slicing axis - 'axial', 'sagittal', or 'coronal'
            all_slices: If True, run inference on ALL slices (including empty ones).
                        If False, only run on slices that have GT labels.

        Returns:
            Dictionary with predictions and metrics:
            {
                'file_name': str,
                'pred_volume': np.ndarray (3D binary prediction),
                'gt_volume': np.ndarray or None (3D ground truth),
                'dice_3d': float or None (3D volume Dice),
                'slice_dices': list of (slice_idx, dice) tuples,
                'num_slices_total': int,
                'num_slices_with_labels': int,
                'num_slices_with_preds': int,
            }
        """
        # Load image volume
        img_nii = nib.load(str(img_file))
        img_volume = img_nii.get_fdata()

        # Load label volume if available
        gt_volume = None
        if label_file is not None and label_file.exists():
            label_nii = nib.load(str(label_file))
            gt_volume = label_nii.get_fdata()
            gt_volume = (gt_volume > 0).astype(np.uint8)

        num_slices = get_num_slices(img_volume, axis)

        # Determine which slices to process
        if all_slices:
            slice_indices = list(range(num_slices))
        else:
            # Only slices with GT labels (same as training)
            slice_indices = []
            if gt_volume is not None:
                for s_idx in range(num_slices):
                    label_slice = extract_slice(gt_volume, s_idx, axis)
                    if label_slice.any():
                        slice_indices.append(s_idx)
            else:
                # No labels available, run on all slices
                slice_indices = list(range(num_slices))

        # Initialize prediction volume (same shape as image volume)
        pred_volume = np.zeros_like(img_volume, dtype=np.uint8)

        slice_dices = []
        num_slices_with_preds = 0

        for s_idx in tqdm(slice_indices, desc=f"  Slicing {img_file.name}", leave=False):
            # Extract 2D slice
            img_slice = extract_slice(img_volume, s_idx, axis)

            # Normalize (same as training)
            img_slice_norm = normalize_slice(img_slice)

            # Convert to RGB PIL Image (same as training)
            img_rgb = np.stack([img_slice_norm] * 3, axis=-1)
            pil_image = PILImage.fromarray(img_rgb, mode='RGB')

            orig_h, orig_w = img_slice.shape

            # Run inference on this slice
            pred_mask = self.predict_slice(pil_image, text_prompt, image_id=s_idx)

            if pred_mask is not None:
                # pred_mask is at original slice resolution (orig_h x orig_w)
                num_slices_with_preds += 1

                # Place prediction back into 3D volume
                if axis == 'axial':
                    pred_volume[:, :, s_idx] = pred_mask.astype(np.uint8)
                elif axis == 'sagittal':
                    pred_volume[s_idx, :, :] = pred_mask.astype(np.uint8)
                else:  # coronal
                    pred_volume[:, s_idx, :] = pred_mask.astype(np.uint8)

            # Compute per-slice Dice if GT available
            if gt_volume is not None:
                gt_slice = extract_slice(gt_volume, s_idx, axis)
                gt_binary = (gt_slice > 0).astype(np.uint8)

                if pred_mask is not None:
                    slice_dice = compute_dice_score(pred_mask, gt_binary)
                else:
                    # No prediction but GT exists => Dice = 0
                    if gt_binary.any():
                        slice_dice = 0.0
                    else:
                        slice_dice = 1.0  # Both empty
                slice_dices.append((s_idx, slice_dice))

        # Compute 3D volume Dice
        dice_3d = None
        if gt_volume is not None:
            dice_3d = compute_dice_score(pred_volume, gt_volume)

        return {
            'file_name': img_file.name,
            'pred_volume': pred_volume,
            'gt_volume': gt_volume,
            'dice_3d': dice_3d,
            'slice_dices': slice_dices,
            'num_slices_total': num_slices,
            'num_slices_with_labels': len(slice_indices) if not all_slices else None,
            'num_slices_with_preds': num_slices_with_preds,
        }

    def predict_folder(
        self,
        images_dir: str,
        labels_dir: Optional[str],
        text_prompt: str,
        axis: str = 'axial',
        num_images: Optional[int] = None,
        all_slices: bool = False,
        output_dir: Optional[str] = None,
        save_nifti: bool = True
    ) -> dict:
        """
        Run inference on a folder of NIfTI volumes.

        Args:
            images_dir: Directory containing NIfTI images
            labels_dir: Directory containing NIfTI labels (optional)
            text_prompt: Text prompt for segmentation
            axis: Slicing axis - 'axial', 'sagittal', or 'coronal'
            num_images: Limit number of volumes to process (None = all)
            all_slices: If True, infer on all slices; if False, only labeled slices
            output_dir: Directory to save results (optional)
            save_nifti: If True, save prediction NIfTI volumes

        Returns:
            Dictionary with overall results:
            {
                'overall_dice': float or None,
                'per_volume_results': list of per-volume result dicts,
                'mean_dice': float or None,
            }
        """
        images_dir = Path(images_dir)
        image_files = sorted(images_dir.glob('*.nii.gz'))[:num_images]

        if len(image_files) == 0:
            raise FileNotFoundError(f"No NIfTI files found in {images_dir}")

        has_labels = labels_dir is not None
        if has_labels:
            labels_dir = Path(labels_dir)
            if not labels_dir.exists():
                print(f"⚠️  Labels directory not found: {labels_dir}")
                has_labels = False

        # Create output directory
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None

        print(f"\n{'='*70}")
        print(f"🏥 NIfTI Folder Inference")
        print(f"{'='*70}")
        print(f"  Images directory: {images_dir}")
        print(f"  Labels directory: {labels_dir if has_labels else 'N/A'}")
        print(f"  Number of volumes: {len(image_files)}")
        print(f"  Slicing axis: {axis}")
        print(f"  Text prompt: '{text_prompt}'")
        print(f"  All slices: {all_slices}")
        print(f"  Output directory: {out_dir if out_dir else 'N/A'}")
        print(f"{'='*70}\n")

        per_volume_results = []
        all_pred_voxels = []
        all_gt_voxels = []

        total_time_start = time.time()

        for vol_idx, img_file in enumerate(image_files):
            print(f"\n📂 [{vol_idx+1}/{len(image_files)}] Processing: {img_file.name}")

            # Find matching label file
            label_file = None
            if has_labels:
                label_file = labels_dir / img_file.name
                if not label_file.exists():
                    print(f"  ⚠️  Label not found: {label_file.name}, skipping Dice for this volume")
                    label_file = None

            vol_time_start = time.time()

            # Run inference on this volume
            result = self.predict_volume(
                img_file=img_file,
                label_file=label_file,
                text_prompt=text_prompt,
                axis=axis,
                all_slices=all_slices
            )

            vol_time_end = time.time()
            vol_time = vol_time_end - vol_time_start

            # Print per-volume results
            print(f"  Total slices: {result['num_slices_total']}")
            if result['num_slices_with_labels'] is not None:
                print(f"  Slices with labels: {result['num_slices_with_labels']}")
            print(f"  Slices with predictions: {result['num_slices_with_preds']}")
            print(f"  Inference time: {vol_time:.1f}s ({vol_time/max(result['num_slices_with_preds'],1):.2f}s/slice)")

            if result['dice_3d'] is not None:
                print(f"  3D Volume Dice: {result['dice_3d']:.4f}")

                # Print per-slice Dice summary
                if len(result['slice_dices']) > 0:
                    slice_dice_values = [d for _, d in result['slice_dices']]
                    print(f"  Per-slice Dice: min={min(slice_dice_values):.4f}, "
                          f"max={max(slice_dice_values):.4f}, "
                          f"mean={np.mean(slice_dice_values):.4f}")

            # Collect voxels for overall Dice computation
            if result['gt_volume'] is not None:
                all_pred_voxels.append(result['pred_volume'].flatten())
                all_gt_voxels.append(result['gt_volume'].flatten())

            # Save prediction NIfTI
            if out_dir is not None and save_nifti:
                # Load original NIfTI to preserve affine and header
                orig_nii = nib.load(str(img_file))
                pred_nii = nib.Nifti1Image(
                    result['pred_volume'].astype(np.uint8),
                    affine=orig_nii.affine,
                    header=orig_nii.header
                )
                pred_path = out_dir / f"pred_{img_file.name}"
                nib.save(pred_nii, str(pred_path))
                print(f"  💾 Saved prediction: {pred_path.name}")

            # Store result summary (without large arrays for JSON serialization)
            result_summary = {
                'file_name': result['file_name'],
                'dice_3d': result['dice_3d'],
                'num_slices_total': result['num_slices_total'],
                'num_slices_with_labels': result['num_slices_with_labels'],
                'num_slices_with_preds': result['num_slices_with_preds'],
                'slice_dices': result['slice_dices'],
                'inference_time_s': vol_time,
            }
            per_volume_results.append(result_summary)

            # Free memory
            del result
            torch.cuda.empty_cache()

        total_time_end = time.time()
        total_time = total_time_end - total_time_start

        # Compute overall Dice (across all voxels from all volumes)
        overall_dice = None
        mean_dice = None

        if len(all_pred_voxels) > 0:
            all_preds = np.concatenate(all_pred_voxels)
            all_gts = np.concatenate(all_gt_voxels)
            overall_dice = compute_dice_score(all_preds, all_gts)

            # Mean Dice across volumes
            volume_dices = [r['dice_3d'] for r in per_volume_results if r['dice_3d'] is not None]
            if len(volume_dices) > 0:
                mean_dice = float(np.mean(volume_dices))

        # Print final summary
        print(f"\n{'='*70}")
        print(f"📊 Final Results")
        print(f"{'='*70}")
        print(f"  Volumes processed: {len(image_files)}")
        print(f"  Total inference time: {total_time/60:.2f} min")

        if has_labels and len(per_volume_results) > 0:
            print(f"\n  Per-Volume 3D Dice Scores:")
            print(f"  {'Volume':<40} {'Dice':>10}")
            print(f"  {'-'*50}")
            for r in per_volume_results:
                dice_str = f"{r['dice_3d']:.4f}" if r['dice_3d'] is not None else "N/A"
                print(f"  {r['file_name']:<40} {dice_str:>10}")

            print(f"  {'-'*50}")
            if mean_dice is not None:
                print(f"  {'Mean Dice (per-volume):':<40} {mean_dice:>10.4f}")
            if overall_dice is not None:
                print(f"  {'Overall Dice (all voxels):':<40} {overall_dice:>10.4f}")

        print(f"{'='*70}")

        # Save results JSON
        if out_dir is not None:
            # Convert slice_dices tuples to serializable format
            results_json = {
                'config': {
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir) if has_labels else None,
                    'text_prompt': text_prompt,
                    'axis': axis,
                    'detection_threshold': self.detection_threshold,
                    'nms_iou_threshold': self.nms_iou_threshold,
                    'resolution': self.resolution,
                    'all_slices': all_slices,
                },
                'overall_dice': overall_dice,
                'mean_dice': mean_dice,
                'total_time_s': total_time,
                'per_volume_results': [
                    {
                        'file_name': r['file_name'],
                        'dice_3d': r['dice_3d'],
                        'num_slices_total': r['num_slices_total'],
                        'num_slices_with_labels': r['num_slices_with_labels'],
                        'num_slices_with_preds': r['num_slices_with_preds'],
                        'inference_time_s': r['inference_time_s'],
                        'slice_dices': [{'slice_idx': s, 'dice': d} for s, d in r['slice_dices']],
                    }
                    for r in per_volume_results
                ]
            }

            results_path = out_dir / "inference_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"\n💾 Results saved to: {results_path}")

        return {
            'overall_dice': overall_dice,
            'mean_dice': mean_dice,
            'per_volume_results': per_volume_results,
        }

    def visualize_volume_slices(
        self,
        img_file: Path,
        label_file: Optional[Path],
        pred_volume: np.ndarray,
        axis: str,
        output_path: str,
        max_slices: int = 16
    ):
        """
        Visualize sample slices from a volume with GT and predictions overlayed.

        Args:
            img_file: Path to NIfTI image
            label_file: Path to NIfTI label (optional)
            pred_volume: 3D prediction volume
            axis: Slicing axis
            output_path: Where to save the visualization
            max_slices: Maximum number of slices to show
        """
        img_volume = nib.load(str(img_file)).get_fdata()
        gt_volume = None
        if label_file is not None and label_file.exists():
            gt_volume = nib.load(str(label_file)).get_fdata()
            gt_volume = (gt_volume > 0).astype(np.uint8)

        num_slices = get_num_slices(img_volume, axis)

        # Select slices to visualize (evenly spaced, prefer labeled slices)
        if gt_volume is not None:
            labeled_slices = []
            for s_idx in range(num_slices):
                gt_slice = extract_slice(gt_volume, s_idx, axis)
                if gt_slice.any():
                    labeled_slices.append(s_idx)

            if len(labeled_slices) > max_slices:
                step = len(labeled_slices) // max_slices
                selected = labeled_slices[::step][:max_slices]
            else:
                selected = labeled_slices
        else:
            step = max(1, num_slices // max_slices)
            selected = list(range(0, num_slices, step))[:max_slices]

        if len(selected) == 0:
            return

        n_cols = min(4, len(selected))
        n_rows = (len(selected) + n_cols - 1) // n_cols
        has_gt = gt_volume is not None
        num_row_groups = 2 if has_gt else 1  # Image, GT (if available), Pred

        fig, axes = plt.subplots(
            n_rows * num_row_groups, n_cols,
            figsize=(4 * n_cols, 3 * n_rows * num_row_groups)
        )
        if n_cols == 1 and n_rows * num_row_groups == 1:
            axes = np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows * num_row_groups == 1:
            axes = axes.reshape(1, -1)

        for i, s_idx in enumerate(selected):
            row = i // n_cols
            col = i % n_cols

            # Extract slices
            img_slice = extract_slice(img_volume, s_idx, axis)
            img_norm = normalize_slice(img_slice)
            pred_slice = extract_slice(pred_volume, s_idx, axis)

            base_row = row * num_row_groups

            if has_gt:
                gt_slice = extract_slice(gt_volume, s_idx, axis)

                # Row 1: GT overlay
                axes[base_row, col].imshow(img_norm, cmap='gray')
                if gt_slice.any():
                    masked_gt = np.ma.masked_where(~(gt_slice > 0), gt_slice.astype(float))
                    axes[base_row, col].imshow(masked_gt, alpha=0.5, cmap='Reds', vmin=0, vmax=1)
                axes[base_row, col].set_title(f'Slice {s_idx}: GT', fontsize=8)
                axes[base_row, col].axis('off')

                # Row 2: Prediction overlay
                pred_row = base_row + 1
            else:
                pred_row = base_row 

            axes[pred_row, col].imshow(img_norm, cmap='gray')
            if pred_slice.any():
                masked_pred = np.ma.masked_where(~(pred_slice > 0), pred_slice.astype(float))
                axes[pred_row, col].imshow(masked_pred, alpha=0.5, cmap='Blues', vmin=0, vmax=1)
            axes[pred_row, col].set_title(f'Slice {s_idx}: Pred', fontsize=8)
            axes[pred_row, col].axis('off')

        # Hide empty subplots
        for i in range(len(selected), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            base_row = row * num_row_groups
            for r_offset in range(num_row_groups):
                if base_row + r_offset < axes.shape[0]:
                    axes[base_row + r_offset, col].axis('off')

        plt.suptitle(f'{img_file.name} - {axis} slices', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  📊 Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + LoRA 3D NIfTI Folder Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic inference:
    python3 infer_sam3_plus_lora_3d_nifti_folder.py \\
        --config configs/full_lora_config_3d_nifti.yaml \\
        --sam3_chk /workspace/sam3.pt \\
        --weights /workspace/output/best_lora_weights.pt \\
        --images_dir /data/imagesTs \\
        --labels_dir /data/labelsTs \\
        --prompt "lesion"

  With output and visualization:
    python3 infer_sam3_plus_lora_3d_nifti_folder.py \\
        --config configs/full_lora_config_3d_nifti.yaml \\
        --sam3_chk /workspace/sam3.pt \\
        --weights /workspace/output/best_lora_weights.pt \\
        --images_dir /data/imagesTs \\
        --labels_dir /data/labelsTs \\
        --prompt "tumor" \\
        --axis axial \\
        --output_dir /workspace/inference_output \\
        --visualize

  Inference on all slices (not just labeled ones):
    python3 infer_sam3_plus_lora_3d_nifti_folder.py \\
        --config configs/full_lora_config_3d_nifti.yaml \\
        --sam3_chk /workspace/sam3.pt \\
        --weights /workspace/output/best_lora_weights.pt \\
        --images_dir /data/imagesTs \\
        --prompt "lesion" \\
        --all_slices \\
        --output_dir /workspace/inference_output
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config_3d_nifti.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--sam3_chk",
        type=str,
        default="/workspace/sam3.pt",
        help="Path to the SAM3 model checkpoint"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to LoRA weights (.pt file)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing NIfTI images (e.g., /data/imagesTs)"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Directory containing NIfTI labels (e.g., /data/labelsTs). "
             "If provided, Dice scores will be computed."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="lesion",
        help='Text prompt for segmentation (e.g., "lesion", "tumor")'
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="axial",
        choices=["axial", "sagittal", "coronal"],
        help="Slicing axis for 3D volumes (default: axial)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Limit the number of NIfTI volumes to process"
    )
    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Run inference on ALL slices (not just labeled ones)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save prediction NIfTI files and results JSON"
    )
    parser.add_argument(
        "--no_save_nifti",
        action="store_true",
        help="Don't save prediction NIfTI volumes (only save results JSON)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images for each volume"
    )
    parser.add_argument(
        "--max_vis_slices",
        type=int,
        default=16,
        help="Maximum number of slices to show in visualization (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )

    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print(f"Using GPU: {args.device}")

    # Initialize model
    inferencer = SAM3LoRANIfTIInference(
        config_path=args.config,
        sam3_checkpoint=args.sam3_chk,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        device="cuda"
    )

    # Run inference on folder
    results = inferencer.predict_folder(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        text_prompt=args.prompt,
        axis=args.axis,
        num_images=args.num_images,
        all_slices=args.all_slices,
        output_dir=args.output_dir,
        save_nifti=not args.no_save_nifti
    )

    # Generate visualizations if requested
    if args.visualize and args.output_dir is not None:
        print(f"\n🎨 Generating visualizations...")
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir) if args.labels_dir else None
        out_dir = Path(args.output_dir)
        vis_dir = out_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_dir.glob('*.nii.gz'))[:args.num_images]

        for vol_result in results['per_volume_results']:
            img_file = images_dir / vol_result['file_name']
            label_file = labels_dir / vol_result['file_name'] if labels_dir else None

            # Load prediction volume
            pred_path = out_dir / f"pred_{vol_result['file_name']}"
            if pred_path.exists():
                pred_volume = nib.load(str(pred_path)).get_fdata()

                vis_path = vis_dir / f"vis_{img_file.stem}.png"
                inferencer.visualize_volume_slices(
                    img_file=img_file,
                    label_file=label_file,
                    pred_volume=pred_volume,
                    axis=args.axis,
                    output_path=str(vis_path),
                    max_slices=args.max_vis_slices
                )
            else:
                print(f"  ⚠️  Prediction NIfTI not found for visualization: {pred_path}")
                print(f"     Re-run without --no_save_nifti to enable visualizations.")

    # Print final Dice scores
    if results['overall_dice'] is not None:
        print(f"\n🎯 Overall Dice (all voxels): {results['overall_dice']:.4f}")
    if results['mean_dice'] is not None:
        print(f"🎯 Mean Dice (per-volume):    {results['mean_dice']:.4f}")


if __name__ == "__main__":
    main()