# セットアップ

## 前提条件

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) がインストール済み
- COCO データセット（または Tiny COCO で動作確認する場合は不要）

## インストール

```bash
git clone <repository-url>
cd ObjectDetection
uv sync
```

`uv sync` で `pyproject.toml` に定義されたすべての依存が `.venv/` に展開されます。  
個別パッケージを追加する場合は `uv add <package>` を使ってください。

## データ準備

### フル COCO データセット

デフォルト設定 (`conf/config.yaml`) では以下のディレクトリ構成を参照します。

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

[COCO 公式サイト](https://cocodataset.org/) からダウンロードして上記パスへ配置してください。

### Tiny COCO（動作確認用）

小規模な Tiny COCO を自動生成するスクリプトが用意されています。

```bash
uv run scripts/make_tiny_coco.py
```

生成先: `data/coco_tiny/`  
詳細は [スクリプト](scripts.md) を参照。

## 動作確認

セットアップが終わったら、Tiny COCO を使って最小実行できます。

```bash
# Tiny COCO を作成してから実行
uv run scripts/make_tiny_coco.py
uv run src/engine.py --config-name config_tiny
```

エラーなく 2 エポック回れば環境は正常です。
