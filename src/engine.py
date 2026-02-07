from core.data_module import CocoDetectionDataModule
from models.ssd import SSD

import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(config: DictConfig):
    # デバイス設定
    device_config = config.device
    if device_config == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA device")
        else:
            device = "cpu"
            print("Using CPU device")
    elif device_config == "cpu":
        device = "cpu"
        print("Using CPU device")
    elif device_config == "cuda":
        device = "cuda"
        print("Using CUDA device")
    elif device_config == "mps":
        device = "mps"
        print("Using MPS (Metal Performance Shaders) device")
    else:
        raise ValueError(f"Unsupported device: {device_config}")

    # モデル選択
    model_name = config.model.name
    if model_name == "ssd":
        model = SSD(config=config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent.parent

    # データモジュルの作成
    data_module = CocoDetectionDataModule(config, project_root)
    data_module.setup()

    # チェックポイント保存の設定
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(project_root / "outputs"),
        filename="ssd-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    # ロガーの設定（TensorBoard優先、失敗時はCSV）
    try:
        logger = TensorBoardLogger(
            save_dir=str(project_root / "lightning_logs"),
            name="ssd_detection",
        )
        print("✓ TensorBoard logger enabled")
    except Exception as e:
        print(f"⚠ TensorBoard logger failed ({e}), using CSV logger")
        logger = CSVLogger(
            save_dir=str(project_root / "lightning_logs"),
            name="ssd_detection",
        )

    # Trainer の設定
    trainer = Trainer(
        max_epochs=config.learning.epochs,
        accelerator=device,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # モデルの学習
    print(f"\n{'=' * 60}")
    print(f"Training SSD for {config.learning.epochs} epochs")
    print(f"Device: {device}")
    print(f"Batch size: {config.learning.batch_size}")
    print(f"{'=' * 60}\n")

    trainer.fit(model, data_module)

    print("\n✓ Training completed!")
    print(f"Logs saved to: {project_root / 'lightning_logs'}")
    print(f"Checkpoints saved to: {project_root / 'outputs'}")


if __name__ == "__main__":
    main()
