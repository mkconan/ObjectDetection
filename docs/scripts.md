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

## sample_dino_v3.py

HuggingFace Transformers の DINOv3 モデルを使って、画像から特徴量を抽出するサンプルスクリプトです。  
`AutoModel` による直接推論と `pipeline` による特徴抽出の 2 つの方法を示します。

### 実行

```bash
uv run scripts/sample_dino_v3.py
```

### 概要

- モデル: `facebook/dinov3-vits16plus-pretrain-lvd1689m`
- COCO 画像を URL から取得し、pooled output の shape を表示
- `image-feature-extraction` パイプラインを使った特徴抽出も実行

### 前提

- インターネット接続（画像ダウンロード・モデルダウンロード）
- `transformers`, `torch` がインストール済みであること（`uv sync` で解決）

## visualize_dino_pca.py

DINOv3 のパッチレベル特徴量を 3 次元 PCA で可視化するスクリプトです。  
PC1→R, PC2→G, PC3→B にマッピングし、元画像と並べて保存します。

### 実行

```bash
uv run scripts/visualize_dino_pca.py
```

### 出力

- `outputs/dino_pca_visualization.png` — 元画像と PCA 可視化の比較画像

### 概要

- モデル: `facebook/dinov3-vits16plus-pretrain-lvd1689m`
- 複数の COCO 画像を URL から取得
- 全画像のパッチ特徴量をまとめて PCA し、グローバルに正規化
- `matplotlib` で 2 列（元画像 / PCA マップ）のグリッドを生成

### 前提

- インターネット接続（画像ダウンロード・モデルダウンロード）
- `transformers`, `torch`, `matplotlib` がインストール済みであること（`uv sync` で解決）
