from abc import ABC, abstractmethod

import torch.nn as nn
from torch.utils.data import DataLoader

from omegaconf import DictConfig


class ModelStrategy(ABC):
    @abstractmethod
    def build(self, config: DictConfig) -> nn.Module:
        """Build the model architecture.

        Args:
            config (DictConfig): _description_

        Returns:
            nn.Module: _description_
        """
        pass

    @abstractmethod
    def train(self, model: nn.Module, train_loader: DataLoader, epochs: int):
        """Train the model using the provided training data.

        Args:
            model (nn.Module): _description_
            train_loader (DataLoader): _description_
            epochs (int): _description_
        """
        pass

    @abstractmethod
    def predict(self, model: nn.Module, input_data):
        """Make predictions based on the input data.

        Args:
            model (nn.Module): _description_
            input_data (_type_): _description_
        """
        pass
