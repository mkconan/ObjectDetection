# 学習実行

エントリーポイントは `src/engine.py`（Hydra main）です。  
`uv run` で実行することで `.venv` の Python を自動的に使用します。

## 基本実行

```bash
uv run src/engine.py
```

デフォルトでは `conf/config.yaml` が読み込まれ、SSD でフル COCO を使って学習が始まります。

## モデル切り替え

Hydra の CLI 上書きでモデルを選択できます。

```bash
# SSD（デフォルト）
uv run src/engine.py model=ssd

# DINOv3
uv run src/engine.py model=dino_v3
```

## 主なパラメータ上書き

```bash
# バッチサイズ
uv run src/engine.py learning.batch_size=16

# エポック数
uv run src/engine.py learning.epochs=100

# 学習率
uv run src/engine.py optimizer.learning_rate=0.0001

# デバイス指定（auto / cpu / cuda / mps）
uv run src/engine.py device=mps
```

複数の上書きは同時に指定できます。

```bash
uv run src/engine.py model=dino_v3 device=cuda learning.epochs=30 learning.batch_size=8
```

## Tiny COCO で最小実行

動作確認や開発中は `conf/config_tiny.yaml` を使うと高速に回せます。

```bash
# 1. Tiny COCO を生成（初回のみ）
uv run scripts/make_tiny_coco.py

# 2. Tiny COCO + 2 エポックで実行
uv run src/engine.py --config-name config_tiny

# モデルやデバイスも上書き可能
uv run src/engine.py --config-name config_tiny model=ssd device=mps
```

`config_tiny.yaml` はデフォルトで `dino_v3` / `batch_size=16` / `epochs=2` の設定です。

## 出力ファイル

| 種別 | 保存先 | 命名規則 |
|---|---|---|
| チェックポイント | `outputs/` | `{model}-epoch={epoch:02d}-val_loss={val_loss:.2f}.ckpt` |
| TensorBoard ログ | `lightning_logs/{model}_detection/` | — |
| MLflow ログ | `mlruns/` | — |

チェックポイントは `val_loss` が低い上位 3 件だけ保持されます（`save_top_k=3`）。

## デバイス自動選択のロジック

`device: auto`（デフォルト）の場合、以下の優先順位でデバイスが選ばれます。

1. MPS（Apple Silicon Mac）が利用可能なら **MPS** を使用
2. CUDA が利用可能なら **CUDA** を使用
3. いずれも不可なら **CPU** を使用

## ロガー起動優先順位

1. MLflow（`conf/config.yaml` に `mlflow` セクションがあれば試みる）
2. TensorBoard（常に追加）
3. CSV（他がすべて失敗した場合のみ）

実際の動作については [ログと実験管理](logging.md) を参照してください。
