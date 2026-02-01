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
