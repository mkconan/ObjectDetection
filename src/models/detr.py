import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50
from torchvision.ops import box_convert, generalized_box_iou

from models.model_base import ModelStrategy

from omegaconf import DictConfig


class DETRBackbone(nn.Module):
    """ResNet-50 backbone for DETR.

    Removes the final average pooling and classification layers so that
    the last convolutional feature map (stride 32) is returned.
    """

    def __init__(self, hidden_dim=256, pretrained=False):
        super().__init__()
        if pretrained:
            from torchvision.models import ResNet50_Weights

            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)

        # Keep all layers except avgpool and fc
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        # Project from 2048 → hidden_dim channels
        self.proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.out_channels = hidden_dim

    def forward(self, x):
        x = self.body(x)
        x = self.proj(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """2D sine/cosine positional encoding as used in the original DETR paper."""

    def __init__(self, hidden_dim=256, temperature=10000, normalize=True):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even"
        self.num_pos_feats = hidden_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        """
        Args:
            x: feature map tensor of shape (B, C, H, W)

        Returns:
            Positional encoding of shape (B, C, H, W)
        """
        B, _, H, W = x.shape
        device = x.device

        # Build integer grid (1-indexed) for y and x axes
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, W).expand(H, W)

        if self.normalize:
            y_embed = y_embed / (H + 1e-6) * self.scale
            x_embed = x_embed / (W + 1e-6) * self.scale

        # Frequency bands: (num_pos_feats,)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Positional embeddings for each axis: (H, W, num_pos_feats)
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        # Concatenate y and x encodings: (H, W, hidden_dim)
        pos = torch.cat([pos_y, pos_x], dim=-1)
        # Reshape to (B, hidden_dim, H, W)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        return pos


class DETR(ModelStrategy):
    """Original DETR (DEtection TRansformer) object detector.

    Implements the architecture from "End-to-End Object Detection with
    Transformers" (Carion et al., 2020).  A ResNet-50 backbone extracts
    features, a Transformer encoder-decoder produces per-query embeddings,
    and separate linear heads predict class logits and bounding boxes.
    Training uses bipartite (Hungarian) matching to assign predictions to
    ground-truth objects.
    """

    def __init__(self, config: DictConfig = None):
        """Initialize the DETR detector.

        Args:
            config (DictConfig, optional): Configuration dictionary.
        """
        super().__init__(config)

        if config is not None:
            num_classes = config.model.get("num_classes", 91)
            num_queries = config.model.get("num_queries", 100)
            hidden_dim = config.model.get("hidden_dim", 256)
            nheads = config.model.get("nheads", 8)
            num_encoder_layers = config.model.get("num_encoder_layers", 6)
            num_decoder_layers = config.model.get("num_decoder_layers", 6)
            pretrained = config.model.get("pretrained", False)
        else:
            num_classes = 91
            num_queries = 100
            hidden_dim = 256
            nheads = 8
            num_encoder_layers = 6
            num_decoder_layers = 6
            pretrained = False

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Backbone: ResNet-50 → hidden_dim feature map
        self.backbone = DETRBackbone(hidden_dim=hidden_dim, pretrained=pretrained)

        # 2D sine positional encoding
        self.pos_encoding = PositionEmbeddingSine(hidden_dim=hidden_dim)

        # Standard PyTorch Transformer (encoder + decoder)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=False,
        )

        # Learnable object query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object class
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, images, targets=None):
        """Forward pass of DETR.

        Args:
            images: List of image tensors or a single batched tensor.
            targets: Unused; kept for API consistency with other detectors.

        Returns:
            Tuple of (outputs_class, outputs_bbox):
                outputs_class: (B, num_queries, num_classes+1) logits
                outputs_bbox:  (B, num_queries, 4) in [cx, cy, w, h] normalized
        """
        if isinstance(images, (list, tuple)):
            x = torch.stack(images, dim=0)
        else:
            x = images

        # Extract features and positional encoding
        features = self.backbone(x)  # (B, hidden_dim, H, W)
        pos = self.pos_encoding(features)  # (B, hidden_dim, H, W)

        B, C, H, W = features.shape

        # Add positional encoding and flatten spatial dims → (H*W, B, C)
        src = (features + pos).flatten(2).permute(2, 0, 1)

        # Object queries: (num_queries, B, hidden_dim)
        tgt = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)

        # Transformer: (num_queries, B, hidden_dim)
        hs = self.transformer(src, tgt)

        # Reshape to (B, num_queries, hidden_dim)
        hs = hs.permute(1, 0, 2)

        outputs_class = self.class_embed(hs)  # (B, num_queries, num_classes+1)
        outputs_bbox = self.bbox_embed(hs).sigmoid()  # (B, num_queries, 4)

        return outputs_class, outputs_bbox

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Args:
            batch: Training batch (images, targets).
            batch_idx: Batch index.

        Returns:
            Total loss value.
        """
        images, targets = batch
        batch_size = len(images)

        images = [img.to(self.device) for img in images]
        batch_targets = self._convert_targets(targets, images)
        batch_targets = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in batch_targets
        ]

        outputs_class, outputs_bbox = self(images)
        losses = self._compute_loss(outputs_class, outputs_bbox, batch_targets)

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
            batch: Validation batch (images, targets).
            batch_idx: Batch index.
        """
        images, targets = batch
        batch_size = len(images)

        images = [img.to(self.device) for img in images]
        batch_targets = self._convert_targets(targets, images)
        batch_targets = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in batch_targets
        ]

        with torch.no_grad():
            outputs_class, outputs_bbox = self(images)
            losses = self._compute_loss(outputs_class, outputs_bbox, batch_targets)

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
            Optimizer instance.
        """
        if self.config is None:
            return torch.optim.Adam(self.parameters(), lr=0.0001)

        optimizer_type = self.config.optimizer.type
        learning_rate = self.config.optimizer.learning_rate

        if optimizer_type == "Adam":
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == "SGD":
            return torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            return torch.optim.Adam(self.parameters(), lr=learning_rate)

    @torch.no_grad()
    def _hungarian_match(self, pred_logits, pred_boxes, gt_labels, gt_boxes):
        """Compute the optimal bipartite matching between predictions and targets.

        Args:
            pred_logits: (num_queries, num_classes+1) class logits
            pred_boxes:  (num_queries, 4) predicted boxes in [cx, cy, w, h]
            gt_labels:   (num_gt,) ground-truth class indices
            gt_boxes:    (num_gt, 4) ground-truth boxes in [cx, cy, w, h]

        Returns:
            Tuple (query_indices, gt_indices) of matched index arrays.
        """
        num_queries = pred_logits.shape[0]
        num_gt = gt_labels.shape[0]

        # Classification cost: negative softmax probability of the target class
        pred_probs = pred_logits.softmax(dim=-1)  # (num_queries, num_classes+1)
        cost_class = -pred_probs[:, gt_labels]  # (num_queries, num_gt)

        # L1 bbox cost
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)  # (num_queries, num_gt)

        # GIoU bbox cost (negate because higher GIoU = better match)
        pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        gt_boxes_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        cost_giou = -generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # (num_queries, num_gt)

        # Combined cost matrix
        cost_matrix = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
        cost_np = cost_matrix.cpu().numpy()

        query_idx, gt_idx = linear_sum_assignment(cost_np)
        return query_idx, gt_idx

    def _compute_loss(self, outputs_class, outputs_bbox, targets):
        """Compute DETR loss using Hungarian matching.

        Args:
            outputs_class: (B, num_queries, num_classes+1) logits
            outputs_bbox:  (B, num_queries, 4) in [cx, cy, w, h] normalized
            targets: List of dicts with 'boxes' (cxcywh normalized) and 'labels'.

        Returns:
            Scalar total loss.
        """
        B = outputs_class.shape[0]
        device = outputs_class.device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_giou_loss = torch.tensor(0.0, device=device)
        num_matched = 0

        for i in range(B):
            pred_logits = outputs_class[i]  # (num_queries, num_classes+1)
            pred_boxes = outputs_bbox[i]  # (num_queries, 4) cxcywh

            gt_boxes = targets[i]["boxes"]  # (num_gt, 4) cxcywh normalized
            gt_labels = targets[i]["labels"]  # (num_gt,)

            num_gt = len(gt_labels)

            # Build no-object target labels for all queries
            no_obj_targets = torch.full(
                (self.num_queries,), self.num_classes, dtype=torch.long, device=device
            )

            if num_gt > 0:
                query_idx, gt_idx = self._hungarian_match(
                    pred_logits.detach(), pred_boxes.detach(), gt_labels, gt_boxes
                )
                query_idx = torch.as_tensor(query_idx, dtype=torch.long, device=device)
                gt_idx = torch.as_tensor(gt_idx, dtype=torch.long, device=device)

                # Assign matched ground-truth labels
                no_obj_targets[query_idx] = gt_labels[gt_idx]

                # BBox losses (only for matched pairs)
                matched_pred = pred_boxes[query_idx]  # (num_matched, 4)
                matched_gt = gt_boxes[gt_idx]  # (num_matched, 4)

                total_bbox_loss = total_bbox_loss + F.l1_loss(matched_pred, matched_gt, reduction="sum")

                matched_pred_xyxy = box_convert(matched_pred, in_fmt="cxcywh", out_fmt="xyxy")
                matched_gt_xyxy = box_convert(matched_gt, in_fmt="cxcywh", out_fmt="xyxy")
                giou = generalized_box_iou(matched_pred_xyxy, matched_gt_xyxy)
                total_giou_loss = total_giou_loss + (1 - giou.diag()).sum()

                num_matched += len(query_idx)

            # Classification loss over all queries (matched + no-object)
            total_cls_loss = total_cls_loss + F.cross_entropy(pred_logits, no_obj_targets)

        # Normalize by total number of matched objects (avoid division by zero)
        normalizer = max(num_matched, 1)
        loss = total_cls_loss / B + (total_bbox_loss + total_giou_loss) / normalizer

        return loss

    @torch.no_grad()
    def predict_boxes(self, images, score_threshold=0.5):
        """Run inference and return filtered detections.

        Args:
            images: List of image tensors (on device).
            score_threshold: Minimum confidence to keep a detection.

        Returns:
            List of dicts, each with:
                - 'boxes': (N, 4) tensor in xyxy pixel coordinates
                - 'labels': (N,) tensor of class indices
                - 'scores': (N,) tensor of confidence scores
        """
        was_training = self.training
        self.eval()

        if isinstance(images, (list, tuple)):
            x = torch.stack(images, dim=0)
        else:
            x = images

        outputs_class, outputs_bbox = self(images)
        # outputs_class: (B, num_queries, num_classes+1)
        # outputs_bbox:  (B, num_queries, 4) cxcywh normalized

        B = outputs_class.shape[0]
        results = []
        for i in range(B):
            probs = outputs_class[i].softmax(dim=-1)  # (num_queries, num_classes+1)
            # Exclude no-object class (last index)
            scores, labels = probs[:, :-1].max(dim=-1)  # (num_queries,)

            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = outputs_bbox[i][keep]  # (N, 4) cxcywh normalized

            # Convert to xyxy pixel coordinates
            img_h, img_w = images[i].shape[-2], images[i].shape[-1]
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            boxes[:, 0::2] *= img_w
            boxes[:, 1::2] *= img_h
            boxes = boxes.clamp(min=0)

            results.append({"boxes": boxes, "labels": labels, "scores": scores})

        if was_training:
            self.train()
        return results

    def _convert_targets(self, targets, images):
        """Convert COCO-format targets to DETR format.

        Converts bounding boxes from COCO [x, y, w, h] pixel coordinates to
        normalized [cx, cy, w, h] in [0, 1] relative to the image dimensions.

        Args:
            targets: List of lists of annotation dicts (COCO format).
            images: List of image tensors used to determine spatial dimensions.

        Returns:
            List of dicts with 'boxes' (cxcywh, normalized) and 'labels'.
        """
        batch_targets = []
        for idx, t_list in enumerate(targets):
            img_h, img_w = images[idx].shape[-2], images[idx].shape[-1]

            if len(t_list) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
            else:
                boxes = [obj["bbox"] for obj in t_list]
                boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

                # COCO xywh (pixel) → cxcywh (normalized)
                boxes = box_convert(boxes, in_fmt="xywh", out_fmt="cxcywh")
                boxes[:, 0] /= img_w  # cx
                boxes[:, 1] /= img_h  # cy
                boxes[:, 2] /= img_w  # w
                boxes[:, 3] /= img_h  # h
                boxes = boxes.clamp(0.0, 1.0)

                labels = [obj["category_id"] for obj in t_list]
                labels = torch.as_tensor(labels, dtype=torch.int64)

            batch_targets.append({"boxes": boxes, "labels": labels})
        return batch_targets
