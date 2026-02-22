# Object Detection Repository

## 概要

2D物体検出モデルを、同じCOCO形式データで切り替えながら学習できるリポジトリです。  
現在は **PyTorch Lightning + Hydra** ベースで学習を実行します。

## 現在の実装状況

- 学習実行: `src/engine.py`（Lightning `Trainer`）
- データ管理: `src/core/data_module.py`（`CocoDetectionDataModule`）
- 実装済みモデル:
  - `ssd`（`src/models/ssd.py`）
  - `dino_v3`（`src/models/dino_v3.py`）
- 設定ファイルは `conf/config.yaml` + `conf/model/*.yaml` を使用
- ログ出力: TensorBoard（失敗時はCSVフォールバック）

- **PyTorch Lightning** による効率的なトレーニング管理
- **Hydra** による柔軟な設定管理
- **COCO データセット** 対応
- 複数の検出モデルをサポート（現在は SSD を実装）
- **MLflow** / TensorBoard / CSV ロギング対応

## セットアップ

```bash
uv sync
```

## データ準備

デフォルト設定（`conf/config.yaml`）では、以下の構成を参照します。

```
data/
└── coco/
  ├── images/
  │   ├── train2017/
  │   └── val2017/
  └── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 学習の実行

### 基本実行

```bash
uv run src/engine.py
```

### Hydra で設定上書き

```bash
# モデル切り替え
uv run src/engine.py model=ssd
uv run src/engine.py model=dino_v3

# バッチサイズ / エポック
uv run src/engine.py learning.batch_size=16
uv run src/engine.py learning.epochs=100

# デバイス（auto / cpu / cuda / mps）
uv run src/engine.py device=mps

# 学習率
uv run src/engine.py optimizer.learning_rate=0.0001
```

### Tiny COCO の作成・確認

```bash
uv run scripts/make_tiny_coco.py
uv run scripts/check_tiny_coco.py
```

### config_tiny.yaml で最小実行

まず Tiny COCO を作成してから、`conf/config_tiny.yaml` を使って短時間で学習を動作確認できます。

```bash
# 1) Tiny COCO を作成
uv run scripts/make_tiny_coco.py

# 2) 最小実行（2エポック）
uv run src/engine.py --config-name config_tiny
```

必要に応じて最小実行時にも上書き可能です。

```bash
# モデル切り替え
uv run src/engine.py --config-name config_tiny model=ssd

# デバイス指定
uv run src/engine.py --config-name config_tiny device=mps
```

## ログと成果物

- チェックポイント: `outputs/`
- Lightningログ: `lightning_logs/`

- `data`: データセットのパス設定
- `learning`: バッチサイズ、エポック数などのトレーニングパラメータ
- `optimizer`: オプティマイザーの種類と学習率
- `device`: 使用するデバイス（auto / cpu / cuda / mps）
- `mlflow`: MLflow トラッキング設定
  - `experiment_name`: 実験名
  - `tracking_uri`: トラッキングサーバーの URI（null の場合は ./mlruns ディレクトリを使用。サーバーを使う場合は "<http://127.0.0.1:5000>" など）
  - `run_name`: 実行名（null の場合は自動生成）

### モデル設定 (`conf/model/ssd.yaml`)

- `name`: モデル名
- `score_threshold`: 検出スコアの閾値
- `input_size`: 入力画像サイズ
- `weights`: 事前学習済みウェイト（COCO_V1 など）

## 出力

トレーニング結果は以下に保存されます:

- **チェックポイント**: `outputs/` ディレクトリ
- **ログ**: `lightning_logs/` ディレクトリ
- **MLflow トラッキング**: `mlruns/` ディレクトリ（MLflow 使用時）

### MLflow でログを確認

MLflow は自動的にトレーニングメトリクスとパラメータを記録します。MLflow UI でログを確認するには:

```bash
# MLflow UI を起動（127.0.0.1:5000）
uv run mlflow ui --port 5000
```

ブラウザで `http://127.0.0.1:5000` にアクセスしてください。

MLflow UI では以下の情報が確認できます:

- トレーニング/検証ロス (train_loss, val_loss)
- ハイパーパラメータ（学習率、バッチサイズなど）
- 実験の比較と可視化
- モデルのバージョン管理

**オプション**: MLflow トラッキングサーバーを使う場合は、`conf/config.yaml` で `mlflow.tracking_uri` を設定してください。

### TensorBoard でログを確認

```bash
uv run tensorboard --logdir lightning_logs --port 6006
```

ブラウザで `http://localhost:6006` を開いて確認できます。

## MLflow の起動方法

このリポジトリには `mlruns/` ディレクトリがあり、MLflow UIで実験履歴を確認できます。

### 1) MLflow を環境に追加

```bash
uv add mlflow
```

### 2) UI を起動

```bash
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

ブラウザで `http://127.0.0.1:5000` を開いて確認してください。

> 補足: 現在の `src/engine.py` は TensorBoard/CSV ロガーを使用しており、MLflow への自動ログ送信は実装されていません。

## 主な設定

### `conf/config.yaml`

- `data.*`: 学習/検証データの画像ディレクトリとアノテーションJSON
- `learning.batch_size`, `learning.epochs`
- `optimizer.type`, `optimizer.learning_rate`
- `device`（`auto` / `cpu` / `cuda` / `mps`）

### `conf/config_tiny.yaml`

- Tiny COCO 用の最小実行設定
- `data.*`: `data/coco_tiny/...` を参照
- `learning.batch_size: 16`
- `learning.epochs: 2`
- 実行コマンド: `uv run src/engine.py --config-name config_tiny`

### `conf/model/ssd.yaml`

- `name`: `ssd`
- `weights`: `COCO_V1` など
- `input_size`, `score_threshold`

### `conf/model/dino_v3.yaml`

- `name`: `dino_v3`
- `image_size`, `num_classes`, `pretrained`

## 開発

```bash
uv run pre-commit install
uv run ruff check --fix
uv run ruff format
```
