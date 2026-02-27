# 設定リファレンス

設定は [Hydra](https://hydra.cc/) で管理されており、`conf/` ディレクトリ以下に YAML として置かれています。

## ファイル構成

```
conf/
├── config.yaml         # フル COCO 用メイン設定
├── config_tiny.yaml    # Tiny COCO 用最小設定
└── model/
    ├── ssd.yaml             # SSD モデル設定
    ├── vit_faster_rcnn.yaml # ViT + Faster R-CNN モデル設定
    └── detr.yaml            # DETR モデル設定
```

## `conf/config.yaml`

```yaml
defaults:
  - _self_
  - model: ssd           # デフォルトモデル

data:
  root: "data/coco/images/train2017"
  ann_file: "data/coco/annotations/instances_train2017.json"
  val_root: "data/coco/images/val2017"
  val_ann_file: "data/coco/annotations/instances_val2017.json"

learning:
  batch_size: 32
  epochs: 50

optimizer:
  type: Adam
  learning_rate: 0.001

device: auto             # auto / cpu / cuda / mps

mlflow:
  experiment_name: "ssd_detection"
  tracking_uri: null     # null → ./mlruns, サーバー利用時は "http://localhost:5000"
  run_name: null         # null → 自動生成
```

### 各キーの説明

| キー | 型 | 説明 |
|---|---|---|
| `data.root` | str | 学習画像のディレクトリ |
| `data.ann_file` | str | 学習アノテーション JSON |
| `data.val_root` | str | 検証画像のディレクトリ |
| `data.val_ann_file` | str | 検証アノテーション JSON |
| `learning.batch_size` | int | バッチサイズ |
| `learning.epochs` | int | 最大エポック数 |
| `optimizer.type` | str | `Adam` / `SGD` |
| `optimizer.learning_rate` | float | 学習率 |
| `device` | str | `auto` / `cpu` / `cuda` / `mps` |
| `mlflow.experiment_name` | str | MLflow 実験名 |
| `mlflow.tracking_uri` | str\|null | MLflow サーバー URI（null で `./mlruns`） |
| `mlflow.run_name` | str\|null | MLflow ラン名（null で自動生成） |

## `conf/config_tiny.yaml`

Tiny COCO 向けの最小設定です。エポック数 2、バッチサイズ 16 で動作確認用に使います。

```yaml
defaults:
  - _self_
  - model: vit_faster_rcnn

data:
  root: "data/coco_tiny/images/train2017"
  ann_file: "data/coco_tiny/annotations/instances_train2017.json"
  val_root: "data/coco_tiny/images/val2017"
  val_ann_file: "data/coco_tiny/annotations/instances_val2017.json"

learning:
  batch_size: 16
  epochs: 2

optimizer:
  type: Adam
  learning_rate: 0.001

device: auto
```

使い方:

```bash
uv run src/engine.py --config-name config_tiny
```

## `conf/model/ssd.yaml`

```yaml
name: ssd
score_threshold: 0.5
input_size: [300, 300]
weights: COCO_V1
```

| キー | 型 | 説明 |
|---|---|---|
| `name` | str | モデル識別子（`model=ssd` で参照） |
| `score_threshold` | float | 検出スコアの閾値 |
| `input_size` | list[int] | 入力画像サイズ `[H, W]` |
| `weights` | str | 事前学習済みウェイト（`COCO_V1` など） |

## `conf/model/vit_faster_rcnn.yaml`

```yaml
name: vit_faster_rcnn
image_size: 224
num_classes: 91
pretrained: false
```

| キー | 型 | 説明 |
|---|---|---|
| `name` | str | モデル識別子 |
| `image_size` | int | 入力画像サイズ（正方形） |
| `num_classes` | int | クラス数（COCO は 91） |
| `pretrained` | bool | 事前学習済みウェイトを使用するか |

## `conf/model/detr.yaml`

```yaml
name: detr
num_queries: 100
num_encoder_layers: 6
num_decoder_layers: 6
hidden_dim: 256
nheads: 8
num_classes: 91
pretrained: false
```

| キー | 型 | 説明 |
|---|---|---|
| `name` | str | モデル識別子 |
| `num_queries` | int | オブジェクトクエリ数 |
| `num_encoder_layers` | int | Transformer エンコーダ層数 |
| `num_decoder_layers` | int | Transformer デコーダ層数 |
| `hidden_dim` | int | Transformer 隠れ層次元数 |
| `nheads` | int | マルチヘッドアテンションのヘッド数 |
| `num_classes` | int | クラス数（COCO は 91） |
| `pretrained` | bool | ResNet-50 バックボーンに事前学習済み重みを使用するか |

## CLI 上書きの例

Hydra はドット記法で任意のキーを上書きできます。

```bash
# 複数キーを同時指定
uv run src/engine.py model=vit_faster_rcnn learning.epochs=30 device=cuda

# Tiny COCO 設定でモデルだけ変える
uv run src/engine.py --config-name config_tiny model=detr
```
