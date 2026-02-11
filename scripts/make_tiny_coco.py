from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).parent.parent
TINY_ROOT = PROJECT_ROOT / "data" / "coco_tiny"
TRAIN_DIR = TINY_ROOT / "images" / "train2017"
VAL_DIR = TINY_ROOT / "images" / "val2017"
ANN_DIR = TINY_ROOT / "annotations"


def _make_image(path: Path, width: int, height: int, box: tuple[int, int, int, int]) -> None:
    """Create a simple image with a colored rectangle."""
    img = Image.new("RGB", (width, height), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(30, 120, 200), width=3)
    img.save(path)


def _build_coco(images: list[dict], annotations: list[dict]) -> dict:
    return {
        "info": {"description": "Tiny COCO for tests"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "box", "supercategory": "shape"},
        ],
    }


def main() -> None:
    random.seed(7)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    def make_split(split_dir: Path, split_name: str, image_count: int, start_id: int) -> None:
        images = []
        annotations = []
        ann_id = start_id * 100
        for i in range(image_count):
            img_id = start_id + i
            width, height = 320, 240
            x = random.randint(20, 120)
            y = random.randint(20, 80)
            w = random.randint(60, 140)
            h = random.randint(50, 120)
            box = (x, y, x + w, y + h)
            file_name = f"{img_id:012d}.jpg"
            _make_image(split_dir / file_name, width, height, box)

            images.append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        coco = _build_coco(images, annotations)
        ann_path = ANN_DIR / f"instances_{split_name}2017.json"
        ann_path.write_text(json.dumps(coco, indent=2))

    make_split(TRAIN_DIR, "train", image_count=4, start_id=1)
    make_split(VAL_DIR, "val", image_count=2, start_id=101)

    print("✓ Tiny COCO dataset created")
    print(f"Train images: {TRAIN_DIR}")
    print(f"Val images:   {VAL_DIR}")
    print(f"Annotations: {ANN_DIR}")


if __name__ == "__main__":
    main()
