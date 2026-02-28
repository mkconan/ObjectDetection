# Object Detection ドキュメント

PyTorch Lightning + Hydra による 2D 物体検出リポジトリです。  
COCO フォーマットのデータセットを使い、複数の検出モデルを切り替えながら学習・評価できます。

## 特徴

- **PyTorch Lightning** — 学習ループ・検証・チェックポイントを自動管理
- **Hydra** — YAML ベースの柔軟な設定管理と CLI 上書き
- **COCO データセット** — 標準フォーマット対応
- **複数モデル対応** — SSD, ViT + Faster R-CNN, DETR を実装済み（今後拡張可能）
- **ロギング** — MLflow / TensorBoard / CSV に対応

## 現在の実装状況

| コンポーネント | ファイル | 状態 |
|---|---|---|
| 学習エントリーポイント | `src/engine.py` | ✅ 実装済み |
| データモジュール | `src/core/data_module.py` | ✅ 実装済み |
| モデル基底クラス | `src/models/model_base.py` | ✅ 実装済み |
| SSD | `src/models/ssd.py` | ✅ 実装済み |
| ViT + Faster R-CNN | `src/models/vit_faster_rcnn.py` | ✅ 実装済み |
| DETR | `src/models/detr.py` | ✅ 実装済み |

## クイックスタート

```bash
# 1. 依存インストール
uv sync

# 2. 学習実行（フルデータ）
uv run src/engine.py

# 3. 最小動作確認（Tiny COCO）
uv run scripts/make_tiny_coco.py
uv run src/engine.py --config-name config_tiny
```

詳しくは [セットアップ](getting-started.md) と [学習実行](training.md) を参照してください。
