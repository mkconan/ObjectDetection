# モデル

このプロジェクトでは **Strategy パターン** を採用しており、`LightningModule` を継承した基底クラス `ModelStrategy` から各モデルを実装します。

## アーキテクチャ概要

```
src/models/
├── model_base.py   # 抽象基底クラス ModelStrategy
├── ssd.py          # SSD 実装
└── dino_v3.py      # DINOv3 実装
```

`engine.py` はモデル名（`config.model.name`）に応じてクラスをインスタンス化し、PyTorch Lightning の `Trainer.fit()` に渡します。

## ModelStrategy（基底クラス）

`src/models/model_base.py` に定義された抽象クラスです。  
新しいモデルを追加する場合はこのクラスを継承して以下の抽象メソッドを実装してください。

| メソッド | 役割 |
|---|---|
| `forward()` | フォワードパス |
| `training_step()` | 学習ステップ（損失の計算・ログ） |
| `validation_step()` | 検証ステップ |
| `configure_optimizers()` | オプティマイザーの設定 |

## SSD

`src/models/ssd.py` / 設定: `conf/model/ssd.yaml`

### 概要

- 骨格: torchvision の `SSD300_VGG16`
- 事前学習重み: `COCO_V1`（デフォルト）
- 入力サイズ: 300×300

### 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `score_threshold` | 0.5 | 推論時の検出スコア閾値 |
| `input_size` | `[300, 300]` | 入力画像サイズ |
| `weights` | `COCO_V1` | 事前学習済みウェイト |

### 学習コマンド

```bash
uv run src/engine.py model=ssd
```

### 内部処理

- `_convert_targets()` で COCO フォーマット → SSD フォーマット（`boxes`, `labels` テンソル）に変換
- `training_step()` で損失を計算し `self.log("train_loss", ...)` でログ
- `validation_step()` で `val_loss` をログ

## DINOv3

`src/models/dino_v3.py` / 設定: `conf/model/dino_v3.yaml`

### 概要

- DINOv3 ベースの検出モデル
- 入力サイズ: 224×224（正方形）
- COCO の 91 クラスに対応

### 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `image_size` | 224 | 入力画像サイズ |
| `num_classes` | 91 | クラス数 |
| `pretrained` | false | 事前学習済み重みを使用するか |

### 学習コマンド

```bash
uv run src/engine.py model=dino_v3
```

## DETR（設定のみ・未実装）

`conf/model/detr.yaml` に設定ファイルのみ存在します。モデル実装は今後の追加予定です。

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `num_queries` | 100 | クエリ数 |
| `num_encoder_layers` | 6 | エンコーダ層数 |
| `num_decoder_layers` | 6 | デコーダ層数 |
| `hidden_dim` | 256 | 隠れ層次元数 |

!!! warning
    `model=detr` を指定するとエラーになります。

## 新しいモデルを追加する

1. `src/models/your_model.py` を作成し `ModelStrategy` を継承
2. `conf/model/your_model.yaml` を作成し `name: your_model` を設定
3. `src/engine.py` のモデル選択ブロックに `elif model_name == "your_model":` を追加
