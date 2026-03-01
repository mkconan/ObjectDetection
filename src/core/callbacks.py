"""Custom PyTorch Lightning callbacks for object detection training."""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Callback
from pytorch_lightning.loggers import MLFlowLogger


# COCO 91-class label names (index 0 = background, 1-90 = object classes)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _get_class_name(label_id: int) -> str:
    """Get class name from COCO label ID, with fallback."""
    if 0 <= label_id < len(COCO_CLASSES):
        return COCO_CLASSES[label_id]
    return f"cls_{label_id}"


def _draw_boxes_on_image(
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor | None,
    gt_labels: torch.Tensor | None,
    pred_boxes: torch.Tensor | None,
    pred_labels: torch.Tensor | None,
    pred_scores: torch.Tensor | None,
) -> Image.Image:
    """Draw GT and predicted bounding boxes on an image.

    Args:
        image_tensor: (C, H, W) float tensor in [0, 1].
        gt_boxes: (M, 4) xyxy pixel coordinates for ground truth.
        gt_labels: (M,) class indices for ground truth.
        pred_boxes: (N, 4) xyxy pixel coordinates for predictions.
        pred_labels: (N,) class indices for predictions.
        pred_scores: (N,) confidence scores for predictions.

    Returns:
        PIL Image with drawn bounding boxes.
    """
    # Convert tensor to PIL Image
    img_np = (image_tensor.cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Draw GT boxes (green)
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box, label in zip(gt_boxes.cpu(), gt_labels.cpu()):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
            name = _get_class_name(int(label))
            text = f"GT: {name}"
            draw.text((x1, max(y1 - 14, 0)), text, fill="lime", font=font)

    # Draw predicted boxes (red)
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box, label, score in zip(
            pred_boxes.cpu(), pred_labels.cpu(), pred_scores.cpu()
        ):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            name = _get_class_name(int(label))
            text = f"{name}: {score:.2f}"
            draw.text((x1, max(y1 - 14, 0)), text, fill="red", font=font)

    return pil_img


def _convert_coco_targets_to_xyxy(targets_list):
    """Convert raw COCO annotation list to xyxy boxes and labels.

    Args:
        targets_list: List of annotation dicts from COCO dataset.

    Returns:
        Tuple of (boxes, labels) tensors in xyxy format.
    """
    if len(targets_list) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.int64)

    boxes = [obj["bbox"] for obj in targets_list]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    # COCO xywh → xyxy
    boxes[:, 2:] += boxes[:, :2]

    labels = [obj["category_id"] for obj in targets_list]
    labels = torch.as_tensor(labels, dtype=torch.int64)

    return boxes, labels


def _find_mlflow_logger(trainer):
    """Find MLFlowLogger from trainer's logger(s).

    Args:
        trainer: PyTorch Lightning Trainer instance.

    Returns:
        MLFlowLogger instance or None.
    """
    logger = trainer.logger
    if isinstance(logger, MLFlowLogger):
        return logger

    # Multiple loggers case
    if hasattr(trainer, "loggers"):
        for lg in trainer.loggers:
            if isinstance(lg, MLFlowLogger):
                return lg

    return None


class BboxVisualizationCallback(Callback):
    """Log detection visualization images to MLflow at each validation epoch end.

    Draws GT bounding boxes (green) and predicted bounding boxes (red) with
    class labels and confidence scores on a subset of validation images.

    Args:
        num_images: Number of images to visualize per epoch.
        score_threshold: Minimum confidence score to display predictions.
    """

    def __init__(self, num_images: int = 4, score_threshold: float = 0.5):
        super().__init__()
        self.num_images = num_images
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        # Skip during sanity check
        if trainer.sanity_checking:
            return

        mlflow_logger = _find_mlflow_logger(trainer)
        if mlflow_logger is None:
            print("  ⚠ No MLflow logger found, skipping bbox visualization")
            return

        # Get validation dataloader
        val_dl = trainer.val_dataloaders
        if val_dl is None:
            print("  ⚠ No val dataloader found, skipping bbox visualization")
            return

        # Get first batch
        try:
            batch = next(iter(val_dl))
        except StopIteration:
            return

        images, targets = batch
        num_to_vis = min(self.num_images, len(images))

        # Move images to model device
        device = pl_module.device
        images_on_device = [img.to(device) for img in images[:num_to_vis]]

        # Run inference
        with torch.no_grad():
            predictions = pl_module.predict_boxes(
                images_on_device, score_threshold=self.score_threshold
            )

        # Get MLflow run ID and client
        run_id = mlflow_logger.run_id
        client = mlflow_logger.experiment

        current_epoch = trainer.current_epoch

        for i in range(num_to_vis):
            # Get GT boxes in xyxy
            gt_boxes, gt_labels = _convert_coco_targets_to_xyxy(targets[i])

            # Draw on image
            pil_img = _draw_boxes_on_image(
                image_tensor=images[i],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=predictions[i]["boxes"],
                pred_labels=predictions[i]["labels"],
                pred_scores=predictions[i]["scores"],
            )

            # Log to MLflow
            artifact_path = f"val_vis/epoch_{current_epoch:03d}_img{i}.png"
            client.log_image(
                run_id=run_id,
                image=pil_img,
                artifact_file=artifact_path,
            )

        print(
            f"  📸 Logged {num_to_vis} bbox visualization images "
            f"(epoch {current_epoch})"
        )
