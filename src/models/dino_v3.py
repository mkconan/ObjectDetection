import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict

from models.model_base import ModelStrategy

from omegaconf import DictConfig


class DINOv3Backbone(nn.Module):
    """Vision Transformer backbone inspired by DINOv3.

    Extracts spatial feature maps from a ViT encoder for use
    with standard object detection heads (e.g., Faster R-CNN).
    Supports positional embedding interpolation to handle
    varying input resolutions.
    """

    def __init__(self, image_size=224, hidden_dim=768, patch_size=16):
        super().__init__()
        vit = vit_b_16(weights=None, image_size=image_size)
        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(hidden_dim, 256, 1)
        self.out_channels = 256

    def forward(self, x):
        n = x.shape[0]

        # Patch embedding
        x = self.conv_proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.reshape(n, self.hidden_dim, h * w).permute(0, 2, 1)

        # Handle positional embedding interpolation for varying input sizes
        num_patches = h * w
        pos_embed_patches = self.encoder.pos_embedding[:, 1:, :]
        expected_patches = pos_embed_patches.shape[1]

        if num_patches != expected_patches:
            orig_size = int(math.sqrt(expected_patches))
            pos_embed_patches = pos_embed_patches.reshape(
                1, orig_size, orig_size, self.hidden_dim
            ).permute(0, 3, 1, 2)
            pos_embed_patches = F.interpolate(
                pos_embed_patches, size=(h, w), mode="bilinear", align_corners=False
            )
            pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(
                1, h * w, self.hidden_dim
            )

        # Prepend class token placeholder
        cls_token = x.new_zeros(n, 1, self.hidden_dim)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embeddings and run through encoder
        pos_embedding = torch.cat(
            [self.encoder.pos_embedding[:, :1, :], pos_embed_patches], dim=1
        )
        x = x + pos_embedding
        x = self.encoder.dropout(x)
        x = self.encoder.layers(x)
        x = self.encoder.ln(x)

        # Remove class token and reshape to spatial feature map
        x = x[:, 1:]
        x = x.permute(0, 2, 1).reshape(n, self.hidden_dim, h, w)

        # Project to detection-friendly channel dimension
        x = self.proj(x)

        return OrderedDict({"0": x})


class DINOv3(ModelStrategy):
    def __init__(self, config: DictConfig = None):
        """Initialize DINOv3-based object detector.

        Uses a Vision Transformer (DINOv3) backbone with a Faster R-CNN
        detection head.

        Args:
            config (DictConfig, optional): Configuration dictionary
        """
        super().__init__(config)

        if config is not None:
            image_size = config.model.get("image_size", 224)
            num_classes = config.model.get("num_classes", 91)
        else:
            image_size = 224
            num_classes = 91

        # Build DINOv3 backbone
        backbone = DINOv3Backbone(image_size=image_size)

        # Anchor generator for single feature map
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        # RoI pooling
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # Build Faster R-CNN with DINOv3 backbone
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size,
        )

    def forward(self, images, targets=None):
        """Forward pass of DINOv3 detector.

        Args:
            images: List of image tensors
            targets: List of target dicts (required during training)

        Returns:
            Loss dict during training, detections during inference
        """
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Args:
            batch: Training batch (images, targets)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        images, targets = batch
        batch_size = len(images)

        # Move images to device
        images = [img.to(self.device) for img in images]

        # Convert targets from COCO format to detection format
        batch_targets = self._convert_targets(targets)
        batch_targets = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in batch_targets
        ]

        # Ensure model is in train mode for loss computation
        self.model.train()

        # Forward pass
        loss_dict = self.model(images, batch_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Log loss
        self.log(
            "train_loss",
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return losses

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning.

        Args:
            batch: Validation batch (images, targets)
            batch_idx: Batch index
        """
        images, targets = batch
        batch_size = len(images)

        # Move images to device
        images = [img.to(self.device) for img in images]

        # Convert targets from COCO format to detection format
        batch_targets = self._convert_targets(targets)
        batch_targets = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in batch_targets
        ]

        # For validation, compute loss without gradients
        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, batch_targets)
            losses = sum(loss for loss in loss_dict.values())

        # Log validation loss
        self.log(
            "val_loss",
            losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        """Configure optimizer based on config.

        Returns:
            Optimizer instance
        """
        if self.config is None:
            return torch.optim.Adam(self.parameters(), lr=0.001)

        optimizer_type = self.config.optimizer.type
        learning_rate = self.config.optimizer.learning_rate

        if optimizer_type == "Adam":
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def _convert_targets(self, targets):
        """Convert targets from COCO format to detection format.

        Args:
            targets: List of list of annotation dicts from COCO

        Returns:
            List of target dicts with 'boxes' and 'labels' tensors
        """
        batch_targets = []
        for t_list in targets:
            if len(t_list) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
                area = torch.zeros(0, dtype=torch.float32)
                iscrowd = torch.zeros(0, dtype=torch.uint8)
            else:
                boxes = [obj["bbox"] for obj in t_list]
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy
                labels = [obj["category_id"] for obj in t_list]
                labels = torch.as_tensor(labels, dtype=torch.int64)
                area = [obj["area"] for obj in t_list]
                area = torch.as_tensor(area, dtype=torch.float32)
                iscrowd = [obj["iscrowd"] for obj in t_list]
                iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

            batch_targets.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "area": area,
                    "iscrowd": iscrowd,
                }
            )
        return batch_targets
