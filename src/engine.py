from core.experiment_base import ExperimentBase
from models.ssd import SSD

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.datasets import CocoDetection

import hydra
from omegaconf import DictConfig
from pathlib import Path


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


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    # デバイス設定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    first_strategy = ExperimentBase(strategy=SSD(), device=device)
    print(config)

    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent.parent

    coco_dataset = CocoDetection(
        root=str(project_root / "data/coco/images/train2017"),
        annFile=str(project_root / "data/coco/annotations/instances_train2017.json"),
        transform=transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
            ]
        ),
    )

    # training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
        coco_dataset, batch_size=config.learning.batch_size, collate_fn=collate_fn_custom
    )
    input_data = None

    # Only run experiment if we have actual data
    first_strategy.run_experiment(config, train_loader, config.learning.epochs, input_data)


if __name__ == "__main__":
    main()
