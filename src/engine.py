from core.data_module import CocoDetectionDataModule
from models.ssd import SSD
from models.dino_v3 import DINOv3

import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, MLFlowLogger


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
    elif model_name == "dino_v3":
        model = DINOv3(config=config)
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
        filename=f"{model_name}" + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    # ロガーの設定（MLflow、TensorBoard、CSV）
    loggers = []
    
    # MLflow logger
    if "mlflow" in config:
        try:
            mlflow_logger = MLFlowLogger(
                experiment_name=config.mlflow.experiment_name,
                tracking_uri=config.mlflow.tracking_uri,
                run_name=config.mlflow.run_name,
            )
            loggers.append(mlflow_logger)
            tracking_info = config.mlflow.tracking_uri if config.mlflow.tracking_uri else "mlruns/"
            print(f"✓ MLflow logger enabled (tracking: {tracking_info})")
        except Exception as e:
            print(f"⚠ MLflow logger failed ({e})")
    else:
        print("ℹ MLflow configuration not found, skipping MLflow logger")
    
    # TensorBoard logger
    try:
        tb_logger = TensorBoardLogger(
            save_dir=str(project_root / "lightning_logs"),
            name=f"{model_name}_detection",
        )
        loggers.append(tb_logger)
        print("✓ TensorBoard logger enabled")
    except Exception as e:
        print(f"⚠ TensorBoard logger failed ({e})")
    
    # CSV logger as fallback when no other loggers are available
    if not loggers:
        csv_logger = CSVLogger(
            save_dir=str(project_root / "lightning_logs"),
            name=f"{model_name}_detection",
        )
        loggers.append(csv_logger)
        print("✓ CSV logger enabled (fallback: no other loggers succeeded)")

    # Trainer の設定
    # Use multiple loggers if more than one, single logger otherwise
    if len(loggers) > 1:
        logger = loggers
    elif len(loggers) == 1:
        logger = loggers[0]
    else:
        logger = None
    
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
    print(f"Training {model_name} for {config.learning.epochs} epochs")
    print(f"Device: {device}")
    print(f"Batch size: {config.learning.batch_size}")
    print(f"{'=' * 60}\n")

    trainer.fit(model, data_module)

    print("\n✓ Training completed!")
    print(f"Logs saved to: {project_root / 'lightning_logs'}")
    print(f"Checkpoints saved to: {project_root / 'outputs'}")


if __name__ == "__main__":
    main()
