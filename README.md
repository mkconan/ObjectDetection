# Object Detection Repository

## 概要

2D 検出モデルの学習および推論を行うためのコードリポジトリです。

データフォーマットは同じだけど、いろんな検出モデルを試したい場合に便利です。

## 特徴

- **PyTorch Lightning** による効率的なトレーニング管理
- **Hydra** による柔軟な設定管理
- **COCO データセット** 対応
- 複数の検出モデルをサポート（現在は SSD を実装）
- **MLflow** / TensorBoard / CSV ロギング対応

## セットアップ

以下のコマンドで必要なライブラリをインストールしてください。

```bash
uv sync
```

## データの準備

COCO データセットを以下の構造で配置してください:

```
data/
├── coco/
│   ├── train2017/        # 訓練画像
│   ├── val2017/          # 検証画像
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
```

## 使い方

### トレーニングの実行

メインのトレーニングスクリプトを実行します:

```bash
uv run python src/engine.py
```

デフォルトでは `conf/config.yaml` の設定が使用されます。

### 設定のカスタマイズ

Hydra を使用して設定を上書きできます:

```bash
# バッチサイズを変更
uv run python src/engine.py learning.batch_size=16

# エポック数を変更
uv run python src/engine.py learning.epochs=100

# デバイスを指定（auto, cpu, cuda, mps）
uv run python src/engine.py device=mps

# 学習率を変更
uv run python src/engine.py optimizer.learning_rate=0.0001
```

### データセットの動作確認

データセットの読み込みテストを行う場合:

```bash
uv run python try.py
```

## 設定ファイル

### メイン設定 (`conf/config.yaml`)

- `data`: データセットのパス設定
- `learning`: バッチサイズ、エポック数などのトレーニングパラメータ
- `optimizer`: オプティマイザーの種類と学習率
- `device`: 使用するデバイス（auto / cpu / cuda / mps）
- `mlflow`: MLflow トラッキング設定
  - `experiment_name`: 実験名
  - `tracking_uri`: トラッキングサーバーの URI（null の場合は ./mlruns ディレクトリを使用。サーバーを使う場合は "http://localhost:5000" など）
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
# MLflow UI を起動（localhost:5000）
uv run mlflow ui --port 5000
```

ブラウザで `http://localhost:5000` にアクセスしてください。

MLflow UI では以下の情報が確認できます:
- トレーニング/検証ロス (train_loss, val_loss)
- ハイパーパラメータ（学習率、バッチサイズなど）
- 実験の比較と可視化
- モデルのバージョン管理

**オプション**: MLflow トラッキングサーバーを使う場合は、`conf/config.yaml` で `mlflow.tracking_uri` を設定してください。

### TensorBoard でログを確認

```bash
uv run tensorboard --logdir lightning_logs
```

## プロジェクト構成

```
.
├── conf/                    # 設定ファイル
│   ├── config.yaml          # メイン設定
│   └── model/               # モデル固有の設定
│       └── ssd.yaml
├── data/                    # データセット
│   └── coco/
├── src/                     # ソースコード
│   ├── engine.py            # メインの実行スクリプト
│   ├── core/                # コアモジュール
│   │   ├── data_module.py   # データローダー
│   │   └── experiment_base.py
│   └── models/              # モデル実装
│       ├── model_base.py
│       └── ssd.py           # SSD モデル
├── outputs/                 # チェックポイント保存先
├── lightning_logs/          # ログ保存先
└── try.py                   # データセット動作確認用
```

## 開発

### Pre-commit フックのセットアップ

コード品質を保つため、pre-commit を使用します:

```bash
uv run pre-commit install
```

### コードフォーマット

```bash
uv run ruff check --fix
uv run ruff format
```
