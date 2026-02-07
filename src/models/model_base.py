from abc import abstractmethod

from pytorch_lightning import LightningModule
from omegaconf import DictConfig


class ModelStrategy(LightningModule):
    """Abstract base class for models using PyTorch Lightning.

    Subclasses should implement training_step(), validation_step(),
    configure_optimizers(), and forward() methods.
    """

    def __init__(self, config: DictConfig = None):
        """Initialize the model strategy.

        Args:
            config (DictConfig, optional): Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Input data

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss value
        """
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning.

        Args:
            batch: Validation batch
            batch_idx: Batch index
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns:
            Optimizer or list of optimizers
        """
        pass
