"""DINOv3 patch feature 3D PCA visualization.

複数の画像のパッチレベル特徴量を3次元PCAで可視化します。
PC1→R, PC2→G, PC3→B にマップして元画像と並べて表示。
"""

import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
PRETRAINED_MODEL = "facebook/dinov3-vits16plus-pretrain-lvd1689m"

IMAGE_URLS = [
    # COCO の猫画像（sample_dino_v3.py と同じ）
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    # Hugging Face ドキュメント用の猫画像
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    # COCO バスの画像
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    # COCO 馬の画像
    "http://images.cocodataset.org/val2017/000000000632.jpg",
]

OUTPUT_PATH = "outputs/dino_pca_visualization.png"

# ──────────────────────────────────────────────
# モデルのロード
# ──────────────────────────────────────────────
print(f"Loading model: {PRETRAINED_MODEL}")
processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)
model = AutoModel.from_pretrained(PRETRAINED_MODEL, device_map="auto")
model.eval()
print(f"Model device: {next(model.parameters()).device}")

# ──────────────────────────────────────────────
# 画像のロードと特徴量抽出
# ──────────────────────────────────────────────
print("\nLoading images and extracting features...")
images = []
patch_features_list = []
patch_grids = []  # (h_patches, w_patches) のリスト

# モデルのパッチサイズとレジスタトークン数を設定から取得
patch_size = model.config.patch_size
n_register_tokens = getattr(model.config, "num_register_tokens", 0)
print(f"Patch size: {patch_size}, Register tokens: {n_register_tokens}")
# last_hidden_state のレイアウト: [CLS, reg_1, ..., reg_n, patch_1, ..., patch_N]
skip = 1 + n_register_tokens  # CLS + レジスタ分をスキップ

for i, url in enumerate(IMAGE_URLS):
    print(f"  [{i+1}/{len(IMAGE_URLS)}] {url.split('/')[-1]}")
    img = load_image(url)
    images.append(img)

    # オリジナル画像サイズをパッチサイズの倍数に切り捨て
    orig_w, orig_h = img.size  # PIL: (width, height)
    input_h = (orig_h // patch_size) * patch_size
    input_w = (orig_w // patch_size) * patch_size

    inputs = processor(
        images=img,
        return_tensors="pt",
        size={"height": input_h, "width": input_w},
        do_resize=True,
    ).to(model.device)
    # pixel_values shape: [1, C, H, W] → H, W からパッチグリッドを計算
    _, c, img_h, img_w = inputs["pixel_values"].shape
    h_patches = img_h // patch_size
    w_patches = img_w // patch_size
    patch_grids.append((h_patches, w_patches))

    with torch.inference_mode():
        outputs = model(**inputs)

    # last_hidden_state: [1, 1+n_register+N_patches, D]
    # CLS (idx0) とレジスタトークン (1..skip-1) をスキップ
    patch_tokens = outputs.last_hidden_state[0, skip:, :]  # [N_patches, D]
    patch_features_list.append(patch_tokens)
    print(f"    original: {orig_w}x{orig_h} → input: {img_w}x{img_h}, patch grid: {h_patches}x{w_patches}, tokens: {patch_tokens.shape}")

# ──────────────────────────────────────────────
# 全画像分のパッチをスタックして PCA (torch)
# ──────────────────────────────────────────────
print("\nRunning PCA on all patch features...")
# torch.pca_lowrank (linalg_qr) は MPS 未対応のため CPU で計算
all_patches = torch.cat(patch_features_list, dim=0).float().cpu()  # [total_N, D]
print(f"  Total patches: {all_patches.shape}")

# torch.pca_lowrank でトップ3主成分を計算
# subtract_mean=True で平均引き算
U, S, V = torch.pca_lowrank(all_patches, q=3, center=True)
# V: [D, 3] — 主成分方向ベクトル
# all_patches @ V で射影できるが、center=True 時は mean を引く必要がある
mean = all_patches.mean(dim=0, keepdim=True)
projected_all = (all_patches - mean) @ V  # [total_N, 3]

# チャンネルごとにグローバル正規化 [0, 1]
def normalize_global(tensor: torch.Tensor) -> torch.Tensor:
    """各チャンネルをグローバルの min/max で [0, 1] に正規化。"""
    t_min = tensor.min(dim=0, keepdim=True).values
    t_max = tensor.max(dim=0, keepdim=True).values
    return (tensor - t_min) / (t_max - t_min + 1e-8)

projected_normalized = normalize_global(projected_all)  # [total_N, 3]

# ──────────────────────────────────────────────
# 各画像のパッチを 2D グリッドに reshape
# ──────────────────────────────────────────────
pca_maps = []
offset = 0
for patch_tokens, (h_patches, w_patches) in zip(patch_features_list, patch_grids):
    n = patch_tokens.shape[0]
    assert h_patches * w_patches == n, (
        f"パッチ数の不一致: grid {h_patches}x{w_patches}={h_patches*w_patches} vs tokens {n}"
    )

    pca_patch = projected_normalized[offset : offset + n]  # [N, 3]
    pca_map = pca_patch.reshape(h_patches, w_patches, 3).cpu().numpy()  # [h, w, 3]
    pca_maps.append(pca_map)
    offset += n

print(f"  PCA map shape per image: {pca_maps[0].shape}")

# ──────────────────────────────────────────────
# matplotlib で可視化
# ──────────────────────────────────────────────
n_images = len(images)
fig, axes = plt.subplots(
    n_images, 2,
    figsize=(10, 5 * n_images),
    gridspec_kw={"wspace": 0.05, "hspace": 0.2},
)

# 1枚のときも 2D 配列になるよう保証
if n_images == 1:
    axes = [axes]

for i, (img, pca_map) in enumerate(zip(images, pca_maps)):
    # 元画像
    axes[i][0].imshow(img)
    axes[i][0].set_title(f"Image {i+1} (original)", fontsize=13)
    axes[i][0].axis("off")

    # PCA 特徴マップ（bilinear で元画像サイズにアップサンプリング済み表示）
    axes[i][1].imshow(
        pca_map,
        interpolation="nearest",
        aspect="equal",
    )
    axes[i][1].set_title(
        f"Image {i+1} — DINOv3 patch features (3D PCA, RGB)",
        fontsize=13,
    )
    axes[i][1].axis("off")

fig.suptitle(
    "DINOv3 Patch Feature Visualization via 3D PCA\n"
    f"Model: {PRETRAINED_MODEL}",
    fontsize=14,
    y=1.001,
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {OUTPUT_PATH}")
plt.show()
