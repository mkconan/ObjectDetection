# ログと実験管理

学習中のメトリクスとハイパーパラメータは、MLflow・TensorBoard・CSV の 3 種類のロガーで記録されます。

## ロガーの種類と起動優先順位

| ロガー | 保存先 | 条件 |
|---|---|---|
| MLflow | `mlruns/` | `conf/config.yaml` に `mlflow` セクションがあれば起動を試みる |
| TensorBoard | `lightning_logs/{model}_detection/` | 常に追加 |
| CSV | `lightning_logs/{model}_detection/` | MLflow / TensorBoard が両方失敗した場合のみ |

## MLflow

### ログ確認（ローカル）

```bash
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

ブラウザで `http://127.0.0.1:5000` にアクセスしてください。

MLflow UI で確認できる情報:

- `train_loss` / `val_loss` の推移
- ハイパーパラメータ（学習率・バッチサイズなど）
- 実験間の比較グラフ
- モデルのバージョン管理

### トラッキングサーバーを使う場合

`conf/config.yaml` の `mlflow.tracking_uri` にサーバーのURLを設定します。

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
```

事前にサーバーを起動しておく必要があります（`uv run mlflow server ...`）。

### 設定項目

| キー | デフォルト | 説明 |
|---|---|---|
| `mlflow.experiment_name` | `"ssd_detection"` | 実験名 |
| `mlflow.tracking_uri` | `null` | `null` なら `./mlruns` を使用 |
| `mlflow.run_name` | `null` | `null` なら自動生成 |

## TensorBoard

```bash
uv run tensorboard --logdir lightning_logs --port 6006
```

ブラウザで `http://localhost:6006` に接続して確認できます。

ログは `lightning_logs/{model}_detection/version_N/` に保存されます。

## 生成物ディレクトリの扱い

以下のディレクトリは学習によって生成されるもので、ドキュメント管理の対象外です。

| ディレクトリ | 内容 |
|---|---|
| `outputs/` | モデルチェックポイント（`.ckpt`） |
| `lightning_logs/` | TensorBoard / CSV ログ |
| `mlruns/` | MLflow 実験記録 |

これらは `.gitignore` に登録して Git 管理から除外することを推奨します。

## チェックポイント

チェックポイントは `outputs/` に自動保存されます。

- 命名規則: `{model}-epoch={epoch:02d}-val_loss={val_loss:.2f}.ckpt`
- 保持数: `val_loss` が低い上位 3 件（`save_top_k=3`）

### チェックポイントから再開

PyTorch Lightning の `Trainer` はチェックポイントからの再開をサポートしています（現在の `engine.py` では明示的な指定が必要です）。
