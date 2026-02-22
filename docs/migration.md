# Lightning 移行履歴

このページでは、手書き学習ループから PyTorch Lightning + Hydra への移行経緯と変更内容を記録します。

## 移行サマリー

ObjectDetection プロジェクトは PyTorch Lightning に正常に移行されました。  
手書きの学習ループは自動化され、検証・チェックポイント・ロギング機能が統合されました。

## アーキテクチャ変更

```
Before:
├── ExperimentBase（Strategy パターン管理）
│   └── strategy.train()（手書きループ）
└── engine.py（Hydra + 手動ループ）

After:
├── LightningModule（ModelStrategy）
│   ├── training_step()
│   ├── validation_step()
│   └── configure_optimizers()
├── LightningDataModule（CocoDetectionDataModule）
│   ├── train_dataloader()
│   └── val_dataloader()
└── engine.py（Trainer で統合）
```

## 主な変更内容

### 1. LightningDataModule の作成（`src/core/data_module.py`）

- COCO Detection データセットを管理する `CocoDetectionDataModule` クラスを実装
- `setup()`, `train_dataloader()`, `val_dataloader()` メソッドを提供
- 既存の `collate_fn_custom()` を保持した正確なバッチ処理

### 2. ModelStrategy を LightningModule ベース化（`src/models/model_base.py`）

- `ModelStrategy` が `LightningModule` を継承
- 抽象メソッド: `forward()`, `training_step()`, `validation_step()`, `configure_optimizers()`
- Strategy パターンとの互換性を維持しつつ Lightning の全機能に対応

### 3. SSD を LightningModule へリファクタ（`src/models/ssd.py`）

- 古い `build()`, `train()`, `predict()` メソッドを廃止
- `_convert_targets()` で COCO → SSD フォーマット変換を内部化
- Lightning のロギング機能（`self.log()`）を活用

### 4. engine.py を Trainer ベース化（`src/engine.py`）

**Before（手書きループ）:**

```python
for epoch in range(epochs):
    for batch in train_loader:
        # Forward, backward, optimize...
```

**After（Trainer）:**

```python
trainer = Trainer(max_epochs=epochs, ...)
trainer.fit(model, data_module)
```

新機能:

- ✅ 自動チェックポイント保存（`outputs/`）
- ✅ TensorBoard / MLflow / CSV ロギング
- ✅ 自動検証ループ
- ✅ デバイス自動選択（CPU / GPU / MPS）
- ✅ 詳細な学習進捗表示

## Before / After 比較

| 項目 | Before | After |
|---|---|---|
| 検証ループ | なし | ✅ 自動実行 |
| チェックポイント | 未実装 | ✅ 自動保存 |
| ログ | print() のみ | ✅ TensorBoard + CSV + MLflow |
| コード行数 | 150+ | 70+ |
| デバイス対応 | 手動 | ✅ 自動 |
| マルチGPU | 未対応 | ✅ ワンラインで対応可能 |
| Early Stopping | なし | ✅ Callback で簡単に追加可能 |

## 今後の拡張例

### Early Stopping

```python
from pytorch_lightning.callbacks import EarlyStopping

callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
trainer = Trainer(callbacks=[checkpoint_callback, callback], ...)
```

### mAP メトリクスの追加

```python
def validation_step(self, batch, batch_idx):
    # ... 既存コード ...
    predictions = self.model([img])
    mAP = compute_coco_metrics(predictions, batch_targets)
    self.log("val_mAP", mAP)
```

### Weights & Biases ロガー

```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(project="object-detection")
trainer = Trainer(logger=logger, ...)
```
