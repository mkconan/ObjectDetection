from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pathlib import Path
from omegaconf import DictConfig


def collate_fn_custom(batch):
    """カスタム collate_fn: サイズ不均一な画像に対応"""
    images = []
    targets = []
    for item in batch:
        images.append(item[0])  # 画像
        targets.append(item[1])  # アノテーション

    # 最初のバッチの画像サイズを基準にリサイズ（簡易的な対応）
    if images:
        ref_size = images[0].shape
        print(f"Batch images info - Count: {len(images)}, Sample shape: {ref_size}")

    return images, targets


class CocoDetectionDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, project_root: Path):
        """
        Initialize the COCO Detection DataModule.

        Args:
            config (DictConfig): Configuration with data paths and batch_size
            project_root (Path): Project root directory path
        """
        super().__init__()
        self.config = config
        self.project_root = project_root

        # Transform configuration
        self.input_size = (300, 300)  # Default SSD input size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

        # Datasets and dataloaders
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        """Set up datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = CocoDetection(
                root=str(self.project_root / self.config.data.root),
                annFile=str(self.project_root / self.config.data.ann_file),
                transform=self.transform,
            )

        if stage == "validate" or stage is None:
            self.val_dataset = CocoDetection(
                root=str(self.project_root / self.config.data.val_root),
                annFile=str(self.project_root / self.config.data.val_ann_file),
                transform=self.transform,
            )

    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.learning.batch_size,
            collate_fn=collate_fn_custom,
            # num_workers=4,  # macOS には worker がない
            shuffle=True,
            # multiprocessing_context="fork",
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.learning.batch_size,
            collate_fn=collate_fn_custom,
            # num_workers=4,  # macOS には worker がない
            shuffle=False,
            # multiprocessing_context="fork",
        )
