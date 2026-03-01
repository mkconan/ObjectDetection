# モデル

このプロジェクトでは **Strategy パターン** を採用しており、`LightningModule` を継承した基底クラス `ModelStrategy` から各モデルを実装します。

## アーキテクチャ概要

```
src/models/
├── model_base.py        # 抽象基底クラス ModelStrategy
├── ssd.py               # SSD 実装
├── vit_faster_rcnn.py   # ViT + Faster R-CNN 実装
└── detr.py              # DETR 実装
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
| `predict_boxes(images, score_threshold)` | **推論専用メソッド**（eval モードで実行し、スコアフィルタ済みの bbox を返す） |

### `predict_boxes()` の戻り値

`List[Dict]` 形式で、各要素は以下のキーを持ちます。

| キー | 型 | 説明 |
|---|---|---|
| `boxes` | `Tensor (N, 4)` | xyxy ピクセル座標の予測 bbox |
| `labels` | `Tensor (N,)` | クラスインデックス |
| `scores` | `Tensor (N,)` | 信頼度スコア |

このメソッドは `BboxVisualizationCallback` から呼び出されます。

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

## ViT + Faster R-CNN

`src/models/vit_faster_rcnn.py` / 設定: `conf/model/vit_faster_rcnn.yaml`

### 概要

- バックボーン: Vision Transformer（ViT-B/16）
- 検出ヘッド: Faster R-CNN
- 入力サイズ: 224×224（正方形）
- COCO の 91 クラスに対応
- バックボーンパラメータは凍結し、検出ヘッドのみ学習

### 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `image_size` | 224 | 入力画像サイズ |
| `num_classes` | 91 | クラス数 |
| `pretrained` | false | 事前学習済み重みを使用するか |

### 学習コマンド

```bash
uv run src/engine.py model=vit_faster_rcnn
```

## DETR

`src/models/detr.py` / 設定: `conf/model/detr.yaml`

### 概要

DETR（DEtection TRansformer）は Carion et al. (2020) が提案したエンドツーエンド物体検出モデルです。  
アンカーや NMS を使わず、Transformer のエンコーダ‐デコーダと Hungarian マッチング損失だけで検出を行います。

- バックボーン: ResNet-50（avgpool / fc を除去、1×1 Conv で `hidden_dim` チャンネルに射影）
- 位置エンコーディング: 2D サイン / コサイン（論文準拠）
- Transformer: 標準の `nn.Transformer`（エンコーダ + デコーダ）
- オブジェクトクエリ: 学習可能な埋め込み（デフォルト 100 個）
- 予測ヘッド: クラスロジットヘッド + 3 層 MLP バウンディングボックスヘッド
- 損失: Hungarian マッチングによる二部グラフマッチング（分類 + L1 + GIoU）

### 設定パラメータ

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `num_queries` | 100 | オブジェクトクエリ数 |
| `num_encoder_layers` | 6 | Transformer エンコーダ層数 |
| `num_decoder_layers` | 6 | Transformer デコーダ層数 |
| `hidden_dim` | 256 | Transformer 隠れ層次元数 |
| `nheads` | 8 | マルチヘッドアテンションのヘッド数 |
| `num_classes` | 91 | クラス数（COCO は 91） |
| `pretrained` | false | ResNet-50 バックボーンに事前学習済み重みを使用するか |

### 学習コマンド

```bash
uv run src/engine.py model=detr
```

複数パラメータの上書き例:

```bash
uv run src/engine.py model=detr learning.epochs=50 optimizer.learning_rate=0.0001 device=cuda
```

## 新しいモデルを追加する

1. `src/models/your_model.py` を作成し `ModelStrategy` を継承
2. `conf/model/your_model.yaml` を作成し `name: your_model` を設定
3. `src/engine.py` のモデル選択ブロックに `elif model_name == "your_model":` を追加
4. `predict_boxes(images, score_threshold=0.5)` を実装して可視化コールバックに対応させる
