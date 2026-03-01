import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from models.model_base import ModelStrategy

from omegaconf import DictConfig


class SSD(ModelStrategy):
    def __init__(self, config: DictConfig = None):
        """Initialize SSD model.

        Args:
            config (DictConfig, optional): Configuration dictionary
        """
        super().__init__(config)

        # Load pre-trained SSD model with specified weights
        if config is not None:
            weights = config.model.weights
            if weights == "COCO_V1":
                self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
            else:
                self.model = ssd300_vgg16(weights=None)
        else:
            self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)

    def forward(self, images, targets=None):
        """Forward pass of SSD model.

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

        # Convert targets from COCO format to SSD format
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

        Note: SSD model in eval mode returns detections instead of loss dict.
        For now, we compute loss by running in train mode without gradient updates.
        A better approach would be to compute mAP metrics on detections.

        Args:
            batch: Validation batch (images, targets)
            batch_idx: Batch index
        """
        images, targets = batch
        batch_size = len(images)

        # Move images to device
        images = [img.to(self.device) for img in images]

        # Convert targets from COCO format to SSD format
        batch_targets = self._convert_targets(targets)
        batch_targets = [
            {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in batch_targets
        ]

        # For validation, we need to compute loss without gradients
        # SSD in eval mode returns predictions, not loss, so we need to use train forward
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
        was_training = self.model.training
        self.model.eval()

        detections = self.model(images)

        results = []
        for det in detections:
            keep = det["scores"] > score_threshold
            results.append({
                "boxes": det["boxes"][keep],
                "labels": det["labels"][keep],
                "scores": det["scores"][keep],
            })

        if was_training:
            self.model.train()
        return results

    def _convert_targets(self, targets):
        """Convert targets from COCO format to SSD format.

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
