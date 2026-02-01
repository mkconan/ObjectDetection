import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from models.model_base import ModelStrategy

from omegaconf import DictConfig


class SSD(ModelStrategy):
    def __init__(self):
        super().__init__()

    def build(self, config: DictConfig) -> nn.Module:
        self.config = config
        # Load pre-trained SSD model with COCO weights
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
        # Optionally, modify the number of classes if needed
        # For COCO, it's 91 classes (80 object classes + background)
        # If you need fewer classes, you can modify the classifier head
        return model

    def train(self, model, train_loader, epochs, device=None):
        if device is None:
            device = torch.device("cpu")

        model.to(device)
        model.train()

        # Use Adam optimizer as per config
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optimizer.learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for images, targets in train_loader:
                # SSD expects list of tensors for images and list of dicts for targets
                images = [img.to(device) for img in images]

                # Convert targets from list of list of dicts to list of dicts
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
                targets = batch_targets

                targets = [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets
                ]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                batch_count += 1

            print(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {epoch_loss / batch_count:.4f}")

    def predict(self, model, input_data, device=None):
        if device is None:
            device = torch.device("cpu")

        model.to(device)
        model.eval()
        with torch.no_grad():
            # input_data should be a list of images
            if isinstance(input_data, list):
                images = [img.to(device) for img in input_data]
            else:
                images = [input_data.to(device)]
            detections = model(images)
        return detections
