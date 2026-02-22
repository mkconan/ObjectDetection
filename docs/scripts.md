# スクリプト

`scripts/` ディレクトリには学習補助のユーティリティスクリプトが入っています。

## make_tiny_coco.py

フル COCO データセットから小規模な Tiny COCO を生成します。  
動作確認・開発中の高速イテレーションに使います。

### 実行

```bash
uv run scripts/make_tiny_coco.py
```

### 出力

```
data/
└── coco_tiny/
    ├── images/
    │   ├── train2017/  （学習用画像: 少数枚をコピー）
    │   └── val2017/    （検証用画像: 少数枚をコピー）
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

### 前提

- `data/coco/` にフル COCO データセットが存在すること

### Tiny COCO を使った学習

```bash
uv run src/engine.py --config-name config_tiny
```

## check_tiny_coco.py

生成した Tiny COCO の内容を確認するスクリプトです。  
画像枚数・アノテーション数・クラス分布などを出力します。

### 実行

```bash
uv run scripts/check_tiny_coco.py
```

### 前提

- `make_tiny_coco.py` で Tiny COCO が生成済みであること

## スクリプト追加時のガイドライン

新しいスクリプトを追加する場合は、以下を守ってください。

- `uv run scripts/your_script.py` で単体実行できるようにする
- 引数が必要なら `argparse` か `Hydra` で受け取る
- 依存パッケージは `uv add` で `pyproject.toml` に追記する
