
#!/usr/bin/env python3
"""
SAM3 LoRA Training Script

Validation Strategy (Following SAM3):
  - During training: Only compute validation LOSS (fast, no metrics)
  - After training: Run validate_sam3_lora.py for full metrics (mAP, cgF1) with NMS

This approach significantly speeds up training by avoiding expensive metric computation
during each epoch, while still monitoring overfitting via validation loss.

Multi-GPU Training:
  Single GPU:
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu

  Multi-GPU with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sam3_lora_native.py --config configs/full_lora_config.yaml --multi-gpu
"""

import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib
import time
# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, save_lora_weights, count_parameters

from torchvision.transforms import v2
import pycocotools.mask as mask_utils  # Required for RLE mask decoding in COCO dataset
from sam3.train.masks_ops import rle_encode  # For encoding masks to RLE format

import nibabel as nib
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Note: Evaluation modules (mAP, cgF1, NMS) are in validate_sam3_lora.py
# Training only computes validation loss, following SAM3's approach


# ============================================================================
# Distributed Training Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def visualize_batch_predictions(input_batch, outputs_list, epoch, batch_idx, save_dir):
    """Visualize batch with GT and predictions overlayed"""
    
    # Access predictions
    with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
        for stage_outputs in outputs_iter:
            pred_outputs = stage_outputs[0]  # First step
            break

    pred_masks = pred_outputs['pred_masks']  # [B, N, H, W]
    pred_logits = pred_outputs['pred_logits']  # [B, N, 1]
    pred_scores = torch.sigmoid(pred_logits).squeeze(-1)  # [B, N]
    
    batch_size = input_batch.img_batch.shape[0]
    img_h, img_w = input_batch.img_batch.shape[2], input_batch.img_batch.shape[3]
    _, axes = plt.subplots(2, batch_size, figsize=(5*batch_size, 10))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    for b in range(batch_size):
        # Get image from img_batch [B, 3, H, W]
        img = input_batch.img_batch[b]  # [3, H, W], normalized
        img = (img * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy().clip(0, 1)
        
        # Row 1: Image + GT
        axes[0, b].imshow(img)
        axes[0, b].set_title(f'Batch {b}: GT')
        axes[0, b].axis('off')
        
        # Overlay GT masks from find_targets
        gt_masks = input_batch.find_targets[0].segments
        if gt_masks is not None and b < gt_masks.shape[0]:
            if gt_masks[b].ndim == 3:
                gt_mask_combined = gt_masks[b].any(dim=0).cpu().numpy()
            elif gt_masks[b].ndim == 2:
                gt_mask_combined = gt_masks[b].cpu().numpy()
            else:
                gt_mask_combined = None
            
            if gt_mask_combined is not None:
                masked_gt = np.ma.masked_where(~gt_mask_combined, gt_mask_combined)
                axes[0, b].imshow(masked_gt, alpha=0.5, cmap='Reds', vmin=0, vmax=1)
        
        # Row 2: Image + Predictions
        axes[1, b].imshow(img)
        axes[1, b].set_title(f'Batch {b}: Pred')
        axes[1, b].axis('off')
        
        # Overlay top-3 predicted masks
        if pred_masks.shape[1] > 0:
            top_scores, top_indices = pred_scores[b].topk(min(3, pred_scores.shape[1]))
            for i, idx in enumerate(top_indices):
                if top_scores[i] > 0.3:
                    pred_mask = torch.sigmoid(pred_masks[b, idx]).detach().unsqueeze(0).unsqueeze(0)
                    pred_mask = torch.nn.functional.interpolate(
                        pred_mask, size=(img_h, img_w), mode='bilinear', align_corners=False
                    ).squeeze().cpu().numpy()
                    binary_mask = pred_mask > 0.5
                    masked_pred = np.ma.masked_where(~binary_mask, binary_mask.astype(float))
                    axes[1, b].imshow(masked_pred, alpha=0.5, cmap='Blues', vmin=0, vmax=1)
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'epoch_{epoch}_batch_{batch_idx}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nFirst batch images saved at: {save_path}")

class NIfTIDataset(Dataset):
    """Dataset class for NIfTI 3D medical images"""
    def __init__(self, images_dir, labels_dir, axis='axial', text_prompt='lesion', num_images=None, all_slices=False):
        """
        Args:
            images_dir: Directory containing NIfTI images (e.g., /data/imagesTr)
            labels_dir: Directory containing NIfTI labels (e.g., /data/labelsTr)
            axis: Slicing axis - 'axial' (default), 'sagittal', or 'coronal'
            text_prompt: Text prompt for all images (e.g., 'lesion', 'tumor')
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.axis = axis
        self.text_prompt = text_prompt
        self.resolution = 1008
        
        # Load NIfTI file lists
        self.image_files = sorted(self.images_dir.glob('*.nii.gz'))[:num_images] 
        
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No NIfTI files found in {images_dir}")
        
        print(f"Loaded NIfTI dataset:")
        print(f"  Loaded 3D-Images: {len(self.image_files)}")
        print(f"  Slicing axis: {axis}")
        print(f"  Text prompt: '{text_prompt}' \n")
        
        
        self.slice_indices = []
        self.volume_shapes = []
        axis_map = {'axial': 2, 'sagittal': 0, 'coronal': 1}

        # Build slice index: (volume_idx, slice_idx) - all slices empty + non-empty 
        for vol_idx, img_file in enumerate(tqdm(self.image_files, desc="Indexing volumes")):
            nii = nib.load(str(img_file))
            volume = nii.get_fdata()
            self.volume_shapes.append(volume.shape)
            num_slices = volume.shape[axis_map[axis]]
            

            if all_slices:
                # Add all slices from this volume
                for slice_idx in range(num_slices):
                    self.slice_indices.append((vol_idx, slice_idx))
            else:
                # Build slice index: (volume_idx, slice_idx) — only non-empty slices
                # Load label volume to check which slices have annotations
                label_file = self.labels_dir / img_file.name
                if not label_file.exists():
                    print(f"  Warning: Label not found for {img_file.name}, skipping volume")
                    continue
                label_volume = nib.load(str(label_file)).get_fdata()
                
                
                non_empty = 0
                for slice_idx in range(num_slices):
                    label_slice = self._extract_slice(label_volume, slice_idx, axis)
                    if label_slice.any():
                        self.slice_indices.append((vol_idx, slice_idx))
                        non_empty += 1
                print(f"  {img_file.name}: {non_empty}/{num_slices} slices with labels")

        if all_slices:
            print(f"  Total 2D slices: {len(self.slice_indices)}")
        else:
            print(f"  Total 2D slices with labels: {len(self.slice_indices)}")

        # Transforms
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(self.resolution, self.resolution), antialias=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _normalize_slice(self, slice_2d):
        """Normalize 2D slice to [0, 255] uint8"""
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
    
    def _extract_slice(self, volume, slice_idx, axis):
        """Extract 2D slice from 3D volume"""
        if axis == 'axial':
            return volume[:, :, slice_idx]
        elif axis == 'sagittal':
            return volume[slice_idx, :, :]
        else:  # coronal
            return volume[:, slice_idx, :]
    
    def _mask_to_boxes_and_segments(self, mask_slice):
        """Convert binary mask to bounding boxes and segmentation masks"""
        # Label connected components
        labeled, num_features = ndimage.label(mask_slice > 0)
        
        if num_features == 0:
            return [], []
        
        boxes = []
        masks = []
        
        for label_id in range(1, num_features + 1):
            component_mask = (labeled == label_id)
            
            # Skip very small components
            if component_mask.sum() < 10:
                continue
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Convert to COCO format [x, y, width, height] normalized
            h, w = mask_slice.shape
            box_xywh = [
                float(x_min / w),
                float(y_min / h),
                float((x_max - x_min + 1) / w),
                float((y_max - y_min + 1) / h)
            ]
            
            boxes.append(box_xywh)
            masks.append(component_mask)
        
        return boxes, masks
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_indices[idx]
        
        # Load image volume
        img_file = self.image_files[vol_idx]
        img_nii = nib.load(str(img_file))
        img_volume = img_nii.get_fdata()
        
        # Load label volume
        label_file = self.labels_dir / img_file.name
        if not label_file.exists():
            raise FileNotFoundError(f"Label not found: {label_file}")
        label_nii = nib.load(str(label_file))
        label_volume = label_nii.get_fdata()
        
        # Extract 2D slices
        img_slice = self._extract_slice(img_volume, slice_idx, self.axis)
        label_slice = self._extract_slice(label_volume, slice_idx, self.axis)
        
        # Normalize image slice
        img_slice_norm = self._normalize_slice(img_slice)
        
        # Convert to RGB PIL Image
        img_rgb = np.stack([img_slice_norm] * 3, axis=-1)
        pil_image = PILImage.fromarray(img_rgb, mode='RGB')
        
        # Store original size
        orig_h, orig_w = img_slice.shape
        
        # Apply transforms to image
        image_tensor = self.transform(pil_image)
        
        # Extract boxes and masks from label
        boxes_xywh, masks = self._mask_to_boxes_and_segments(label_slice)
        
        # Create SAM3 Objects
        objects = []
        for box_xywh, mask in zip(boxes_xywh, masks):
            # Convert box to cxcywh format (NORMALIZED to [0,1])
            x, y, bw, bh = box_xywh
            cx = x + bw / 2
            cy = y + bh / 2
            box_tensor = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
            
            # Resize mask from original resolution to model resolution (1008x1008)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_t = torch.nn.functional.interpolate(
                mask_t,
                size=(self.resolution, self.resolution),
                mode="nearest"
            )
            segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor
            
            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=len(objects),
                segment=segment
            )
            objects.append(obj)
        
        # If no objects, create empty list
        if len(objects) == 0:
            objects = []
        
        # Create SAM3 Image
        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )
        
        # Create FindQuery
        query = FindQueryLoaded(
            query_text=self.text_prompt,
            image_id=0,
            object_ids_output=[obj.object_id for obj in objects],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx,
                original_image_id=idx,
                original_category_id=1,
                original_size=(orig_h, orig_w),
                object_id=-1,
                frame_index=-1
            )
        )
        
        # Create Datapoint
        return Datapoint(find_queries=[query], images=[image_obj], raw_images=[pil_image])


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.3):
    """
    Merge overlapping masks that likely represent the same object.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.3)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format for evaluation.

    OPTIMIZATION: Keep masks at native model output resolution (288×288)
    GT is downsampled to match, so no upsampling needed!

    Args:
        predictions_list: List of prediction dictionaries from the model
        image_ids: List of image IDs corresponding to predictions
        resolution: Mask resolution for evaluation (default: 288, model's native output)
        score_threshold: Minimum score threshold for predictions
        merge_overlaps: Whether to merge overlapping predictions (default: True)
        iou_threshold: IoU threshold for merging overlaps (default: 0.3)
        debug: Print debug information

    Returns:
        List of prediction dictionaries in COCO format
    """
    coco_predictions = []
    pred_id = 0

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Extract predictions
        logits = preds['pred_logits']  # [num_queries, 1]
        boxes = preds['pred_boxes']    # [num_queries, 4]
        masks = preds['pred_masks']    # [num_queries, H, W]

        scores = torch.sigmoid(logits).squeeze(-1)  # [num_queries]

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:  # Debug first image only
            print(f"  Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")

        # Convert masks to binary (apply sigmoid first, then threshold)
        binary_masks = (torch.sigmoid(masks) > 0.5).cpu()

        # Merge overlapping predictions to avoid over-segmentation penalty
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"  Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        # Encode masks to RLE (at native resolution - much faster!)
        if len(binary_masks) > 0:
            # Check if masks have content
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"  Mask shape: {binary_masks.shape}")
                print(f"  Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [cx, cy, w, h] to [x, y, w, h] in pixel coordinates
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                coco_predictions.append({
                    'image_id': int(img_id),
                    'category_id': 1,  # Single category for instance segmentation
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                })
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from SimpleSAM3Dataset.

    OPTIMIZATION: Downsample GT masks to match prediction resolution (288×288)
    instead of upsampling predictions to 1008×1008. Much faster!

    Args:
        dataset: SimpleSAM3Dataset instance
        image_ids: Optional list of specific image IDs to include
        mask_resolution: Resolution to downsample masks to (default: 288 to match model output)

    Returns:
        Dictionary in COCO format
    """
    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    # Scale factor for boxes (masks will be at mask_resolution, boxes scaled accordingly)
    scale_factor = mask_resolution / dataset.resolution

    for idx in indices:
        # Add image entry at mask resolution
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True  # Required for cgF1 evaluation
        })

        # Get datapoint
        datapoint = dataset[idx]

        # Add annotations
        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at mask_resolution
            cx, cy, bw, bh = (obj.bbox * mask_resolution).tolist()
            x, y, w, h = cx - bw / 2, cy - bh / 2, bw, bh

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            # Add segmentation if available - downsample to mask_resolution
            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    return coco_gt


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        # Process on GPU then immediately move to CPU to save memory
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Free GPU memory immediately after upsampling
        del masks_sigmoid, masks_upsampled
        torch.cuda.empty_cache()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Convert normalized CxCyWH box to COCO [x, y, w, h] at original size
            cx, cy, bw, bh = obj.bbox.tolist()
            w = bw * orig_w
            h = bh * orig_h
            x = cx * orig_w - w / 2
            y = cy * orig_h - h / 2

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


class SAM3TrainerNative:
    def __init__(self, config_path, sam3_chkpoint_dir, init_lora_weights, output_dir, num_epochs, num_images, all_slices, save_model_every_epoch, multi_gpu=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            self.output_dir = output_dir
            self.num_epochs = num_epochs
            self.train_num = num_images[0]
            self.val_num = num_images[1]
            self.all_slices = all_slices
            self.save_model_every_epoch = save_model_every_epoch

        # Multi-GPU setup
        self.multi_gpu = multi_gpu
        self.local_rank = 0
        self.world_size = 1

        if self.multi_gpu:
            self.local_rank = setup_distributed()
            self.world_size = get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            print_rank0(f"Multi-GPU training enabled with {self.world_size} GPUs")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Model
        print_rank0("Building/Loading SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            checkpoint_path=sam3_chkpoint_dir, 
            load_from_HF=False,
            eval_mode=False
            )

        # Apply LoRA
        print_rank0("Applying LoRA...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
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
        if init_lora_weights is not None:
            print(f"💾 Loading LoRA weights from {init_lora_weights}...")
            load_lora_weights(self.model, init_lora_weights)

        stats = count_parameters(self.model)
        print_rank0(f"Trainable params: {stats['trainable_parameters']:,} / {stats['total_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        self.model.to(self.device)

        # Wrap model with DDP if multi-GPU
        if self.multi_gpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # Frozen params (requires_grad=False) don't need this flag
            )
            print_rank0(f"Model wrapped with DistributedDataParallel")

        # Store reference to unwrapped model for accessing custom methods
        self._unwrapped_model = self.model.module if self.multi_gpu else self.model

        # Optimizer
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Matcher & Loss
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )

        # Create loss functions with correct weights (from original SAM3 training config)
        # Note: These weights are for mask-based training
        loss_fns = [
            Boxes(weight_dict={
                "loss_bbox": 5.0,
                "loss_giou": 2.0
            }),
            IABCEMdetr(
                pos_weight=10.0,
                weight_dict={
                    "loss_ce": 20.0,
                    "presence_loss": 20.0
                },
                pos_focal=False,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pad_n_queries=200,
            ),
            Masks(
                weight_dict={
                    "loss_mask": 200.0,  # Much higher weight for mask loss!
                    "loss_dice": 10.0
                },
                focal_alpha=0.25,
                focal_gamma=2.0,
                compute_aux=False
            )
        ]

        # Create one-to-many matcher for auxiliary outputs
        o2m_matcher = BinaryOneToManyMatcher(
            alpha=0.3,
            threshold=0.4,
            topk=4
        )

        # Use Sam3LossWrapper for proper loss computation
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=loss_fns,
            matcher=self.matcher,
            o2m_matcher=o2m_matcher,
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",  # Use local normalization (no distributed training)
            normalize_by_valid_object_num=False,
        )
        
    def train(self):
        # Get data directory from config (should point to directory containing train/valid folders)
        data_dir = self.config["training"]["data_dir"]
        text_prompt = self.config["training"]["text_prompt"]
        axis_3rd_dimension = self.config["training"]["3rd_dimension_axis"]
        
        # Load datasets using COCO format
        print_rank0(f"\nLoading training data from {data_dir}...")

        # NIfTI dataset paths
        imagesTr_dir = os.path.join(data_dir, "imagesTr")
        labelsTr_dir = os.path.join(data_dir, "labelsTr")

        train_ds = NIfTIDataset(
            images_dir=imagesTr_dir,
            labels_dir=labelsTr_dir,
            axis=axis_3rd_dimension,  # Change to 'sagittal' or 'coronal' if needed (axial)
            text_prompt=text_prompt,  # Change to your target object name
            num_images=self.train_num,
            all_slices=self.all_slices
        )

        # Check if validation data exists
        has_validation = False
        val_ds = None

        try:
            print_rank0(f"\nLoading validation data from {data_dir}...")
            imagesTs_dir = os.path.join(data_dir, "imagesTs")
            labelsTs_dir = os.path.join(data_dir, "labelsTs")
            val_ds = NIfTIDataset(
                images_dir=imagesTs_dir,
                labels_dir=labelsTs_dir,
                axis=axis_3rd_dimension,  # Change to 'sagittal' or 'coronal' if needed (axial)
                text_prompt=text_prompt,  # Change to your target object name
                num_images=self.val_num,
                all_slices=self.all_slices
            )

            if len(val_ds) > 0:
                has_validation = True
                print_rank0(f"Found validation data: {len(val_ds)} 2D-slices.")
            else:
                print_rank0(f"Validation dataset is empty.")
                val_ds = None
        except Exception as e:
            print_rank0(f"Could not load validation data: {e}")
            val_ds = None

        if not has_validation:
            val_ds = None

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        # Create samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.multi_gpu:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True
            )
            if has_validation:
                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=get_rank(),
                    shuffle=False
                )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),  # Only shuffle if not using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.config["training"].get("num_workers", 0),
            pin_memory=True
        )

        if has_validation:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=self.config["training"].get("num_workers", 0),
                pin_memory=True
            )
        else:
            val_loader = None

        self.model.train()

        # Weights from a standard SAM config roughly
        weight_dict = {
            "loss_ce": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0
        }

        # epochs = self.config["training"]["num_epochs"]
        epochs = self.num_epochs
        best_val_loss = float('inf')
        print_rank0(f"Starting training for {epochs} epochs...")

        if has_validation:
            print_rank0(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        else:
            print_rank0(f"Training samples: {len(train_ds)}")
            print_rank0("⚠️  No validation data found - training without validation")

        if self.multi_gpu:
            print_rank0(f"Effective batch size: {self.config['training']['batch_size']} x {self.world_size} = {self.config['training']['batch_size'] * self.world_size}")

        # Helper to move BatchedDatapoint to device
        def move_to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, list):
                return [move_to_device(x, device) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(move_to_device(x, device) for x in obj)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                for field in obj.__dataclass_fields__:
                    val = getattr(obj, field)
                    setattr(obj, field, move_to_device(val, device))
                return obj
            return obj

        # Create output directory
        # out_dir = Path(self.config["output"]["output_dir"])
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            # Set epoch for distributed sampler (required for proper shuffling)
            if self.multi_gpu and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Track training losses for this epoch
            train_losses = []

            # Only show progress bar on rank 0
            pbar = tqdm(enumerate(train_loader), desc=f"Train Epoch {epoch+1}", disable=not is_main_process())
            train_time_start = time.time()
            for batch_idx, batch_dict in pbar:
                input_batch = batch_dict["input"]

                # Move to device
                input_batch = move_to_device(input_batch, self.device)

                # Forward pass
                # outputs_list is SAM3Output, we need to pass the whole thing to loss_wrapper
                outputs_list = self.model(input_batch)
                
                # Prepare targets for loss
                # input_batch.find_targets is a list of BatchedFindTarget (one per stage)
                find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                # Visualize first batch of each epoch
                if batch_idx == 0 and is_main_process():
                    vis_dir = out_dir / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    visualize_batch_predictions(input_batch, outputs_list, epoch, batch_idx, vis_dir)

                # Move targets to device
                for targets in find_targets:
                    for k, v in targets.items():
                        if isinstance(v, torch.Tensor):
                            targets[k] = v.to(self.device)

                # Add matcher indices to outputs (required by Sam3LossWrapper)
                # Use SAM3Output.iteration_mode to properly iterate over outputs
                with SAM3Output.iteration_mode(
                    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                ) as outputs_iter:
                    for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                        # stage_targets is a single target dict, replicate for all steps
                        stage_targets_list = [stage_targets] * len(stage_outputs)
                        for outputs, targets in zip(stage_outputs, stage_targets_list):
                            # Compute indices for main output
                            outputs["indices"] = self.matcher(outputs, targets)

                            # Also add indices to auxiliary outputs if they exist
                            if "aux_outputs" in outputs:
                                for aux_out in outputs["aux_outputs"]:
                                    aux_out["indices"] = self.matcher(aux_out, targets)

                # Compute loss using Sam3LossWrapper
                # This handles num_boxes calculation and proper weighting
                loss_dict = self.loss_wrapper(outputs_list, find_targets)

                # Extract total loss
                total_loss = loss_dict[CORE_LOSS_KEY]

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Track training loss
                train_losses.append(total_loss.item())
                pbar.set_postfix({"loss": total_loss.item()})

            # Calculate average training loss for this epoch
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            train_time_end = time.time()
            print_rank0(f'\nTrain time epoch {epoch+1}/{epochs} for {len(train_loader.dataset)} slices: {(train_time_end-train_time_start)/60:.2} min\n')
            
            # Validation (only compute loss - no metrics, like SAM3)
            if has_validation and val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc=f"Validation", disable=not is_main_process())
                    val_time_start = time.time()

                    for batch_dict in val_pbar:
                        input_batch = batch_dict["input"]
                        input_batch = move_to_device(input_batch, self.device)

                        # Forward pass
                        outputs_list = self.model(input_batch)

                        # Prepare targets
                        find_targets = [self._unwrapped_model.back_convert(target) for target in input_batch.find_targets]

                        # Move targets to device
                        for targets in find_targets:
                            for k, v in targets.items():
                                if isinstance(v, torch.Tensor):
                                    targets[k] = v.to(self.device)

                        # Add matcher indices to outputs (required by Sam3LossWrapper)
                        with SAM3Output.iteration_mode(
                            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
                        ) as outputs_iter:
                            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                                stage_targets_list = [stage_targets] * len(stage_outputs)
                                for outputs, targets in zip(stage_outputs, stage_targets_list):
                                    outputs["indices"] = self.matcher(outputs, targets)
                                    if "aux_outputs" in outputs:
                                        for aux_out in outputs["aux_outputs"]:
                                            aux_out["indices"] = self.matcher(aux_out, targets)

                        # Compute loss using Sam3LossWrapper
                        loss_dict = self.loss_wrapper(outputs_list, find_targets)
                        total_loss = loss_dict[CORE_LOSS_KEY]

                        val_losses.append(total_loss.item())
                        val_pbar.set_postfix({"val_loss": total_loss.item()})

                avg_val_loss = sum(val_losses) / len(val_losses)

                val_time_end = time.time()
                print_rank0(f'\nValidation time epoch {epoch+1}/{epochs} for {len(val_loader.dataset)} slices: {(val_time_end-val_time_start)/60:.2} min\n')
                
                # Synchronize val_loss across all processes for consistent best model selection
                if self.multi_gpu:
                    val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                print_rank0(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

                # Save models based on validation loss (only on rank 0)
                if is_main_process():
                    # Get underlying model from DDP wrapper
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if self.save_model_every_epoch:
                        save_lora_weights(model_to_save, str(out_dir / f"epoch_{epoch+1:03d}_lora_weights.pt"))
                    

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_lora_weights(model_to_save, str(out_dir / "best_lora_weights.pt"))
                        print(f"✓ New best model saved (val_loss: {avg_val_loss:.6f})")

                    # Log to file
                    with open(out_dir / "val_stats.json", "a") as f:
                        f.write(json.dumps({
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss
                        }) + "\n")

                torch.cuda.empty_cache()

                # Back to training mode
                self.model.train()
            else:
                # No validation - just save model each epoch (only on rank 0)
                if is_main_process():
                    model_to_save = self.model.module if self.multi_gpu else self.model
                    save_lora_weights(model_to_save, str(out_dir / "last_lora_weights.pt"))

                    if self.save_model_every_epoch:
                        save_lora_weights(model_to_save, str(out_dir / f"epoch_{epoch+1:03d}_lora_weights.pt"))

        # Synchronize before final save
        if self.multi_gpu:
            dist.barrier()

        # Final save (only on rank 0)
        if is_main_process():
            if has_validation:
                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (best validation loss)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"{'='*80}")
            else:
                # If no validation, copy last to best
                import shutil
                last_path = out_dir / "last_lora_weights.pt"
                best_path = out_dir / "best_lora_weights.pt"
                if last_path.exists():
                    shutil.copy(last_path, best_path)

                print(f"\n{'='*80}")
                print(f"✅ Training complete!")
                print(f"{'='*80}")
                print(f"\nModels saved to {out_dir}:")
                print(f"  - best_lora_weights.pt (copy of last epoch)")
                print(f"  - last_lora_weights.pt (last epoch)")
                print(f"\nℹ️  No validation data - consider adding data/valid/ for better model selection")
                print(f"{'='*80}")

        # Cleanup distributed training
        if self.multi_gpu:
            cleanup_distributed()

def launch_distributed_training(args):
    """Launch training with multiple GPUs using torchrun subprocess."""
    import subprocess
    import sys

    devices = args.device
    num_gpus = len(devices)
    device_str = ",".join(map(str, devices))

    print(f"Launching distributed training on GPUs: {devices}")
    print(f"Number of processes: {num_gpus}")

    # Build the command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--master_port", str(args.master_port),
        sys.argv[0],  # This script
        "--config", args.config,
        "--device", *map(str, devices),
        "--_launched_by_torchrun"  # Internal flag to indicate we're in subprocess
    ]

    # Set environment variable for visible devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_str

    # Run the subprocess
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAM3 with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU (default GPU 0):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml

  Single GPU (specific GPU):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 1

  Multi-GPU (GPUs 0 and 1):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1

  Multi-GPU (GPUs 0, 2, 3):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 2 3

  Multi-GPU (all 4 GPUs):
    python train_sam3_lora_native.py --config configs/full_lora_config.yaml --device 0 1 2 3
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_lora_config_3d_nifti.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--sam3_chkdir",
        type=str,
        default="/workspace/sam3.pt",
        help="Path to the sam3 model saved locally"
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default=None,
        help="Path to the saved LoRA weights locally"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output",
        help="Path to the output folder"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Limit the number of epochs"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        nargs=2,
        default=[None, None],
        help="Limits the number of the nifit images loaded for [train,validation]"
    )
    parser.add_argument(
        "--all_slices",
        action="store_true",
        help="Run inference and train on ALL slices (not just labeled ones)"
    )
    parser.add_argument(
        "--save_model_every_epoch",
        action="store_true",
        help="Save the LoRA model every epoch"
    )
    parser.add_argument(
        "--device",
        type=int,
        nargs="+",
        default=[0],
        help="GPU device ID(s) to use. Single value for single GPU, multiple values for multi-GPU. "
             "Example: --device 0 (single GPU), --device 0 1 2 (3 GPUs)"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training (default: 29500)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set automatically by torchrun)"
    )
    parser.add_argument(
        "--_launched_by_torchrun",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden argument for internal use
    )
    args = parser.parse_args()

    # Determine if multi-GPU training is requested
    num_devices = len(args.device)
    is_torchrun_subprocess = args._launched_by_torchrun or "LOCAL_RANK" in os.environ

    if num_devices > 1 and not is_torchrun_subprocess:
        # Multi-GPU requested but not yet in torchrun - launch it
        launch_distributed_training(args)
    else:
        # Single GPU or already in torchrun subprocess
        multi_gpu = num_devices > 1 and is_torchrun_subprocess

        if not multi_gpu and num_devices == 1:
            # Single GPU mode - set the device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[0])
            print(f"Using single GPU: {args.device[0]}")

        trainer = SAM3TrainerNative(args.config, args.sam3_chkdir, args.init_lora_weights, args.output_dir, args.num_epochs, args.num_images, args.all_slices, args.save_model_every_epoch, multi_gpu=multi_gpu)
        trainer.train()
