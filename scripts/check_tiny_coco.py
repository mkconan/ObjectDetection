from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.data_module import CocoDetectionDataModule


def main() -> None:
    config = OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml")
    config.data.root = "data/coco_tiny/images/train2017"
    config.data.ann_file = "data/coco_tiny/annotations/instances_train2017.json"
    config.data.val_root = "data/coco_tiny/images/val2017"
    config.data.val_ann_file = "data/coco_tiny/annotations/instances_val2017.json"
    config.learning.batch_size = 2

    data_module = CocoDetectionDataModule(config, PROJECT_ROOT)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("✓ Train batch ready")
    print(f"Train images: {len(train_batch[0])}")
    print(f"Train targets: {len(train_batch[1])}")

    print("✓ Val batch ready")
    print(f"Val images: {len(val_batch[0])}")
    print(f"Val targets: {len(val_batch[1])}")


if __name__ == "__main__":
    main()
