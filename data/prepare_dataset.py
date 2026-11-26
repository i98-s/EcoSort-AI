import hashlib
import zipfile
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download

TARGET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
RAW_DIR = Path("data/raw_trashnet")
OUTPUT_DIR = Path("data/cleaned_dataset")


def get_image_hash(pil_img: Image.Image) -> str:
    return hashlib.md5(pil_img.tobytes()).hexdigest()


def curate_and_clean():
    print("Downloading TrashNet archive from Hugging Face Hub...")
    zip_path = hf_hub_download(
        repo_id="garythung/trashnet",
        filename="dataset-resized.zip",
        repo_type="dataset",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    for cls_name in TARGET_CLASSES:
        (OUTPUT_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    stats = {"saved": 0, "corrupted": 0, "duplicates": 0}

    # Walk extracted folders and match parent folder name to TARGET_CLASSES
    for img_path in sorted(RAW_DIR.rglob("*")):
        if not img_path.is_file() or img_path.name.startswith("."):
            continue

        cls_name = img_path.parent.name.lower().strip()
        if cls_name not in TARGET_CLASSES:
            continue

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                rgb_img = im.convert("RGB")
        except Exception:
            stats["corrupted"] += 1
            continue

        img_hash = get_image_hash(rgb_img)
        if img_hash in seen_hashes:
            stats["duplicates"] += 1
            continue
        seen_hashes.add(img_hash)

        save_path = OUTPUT_DIR / cls_name / f"{cls_name}_{stats['saved']:05d}.jpg"
        rgb_img.save(save_path, format="JPEG", quality=95)
        stats["saved"] += 1

    print(
        f"Done! Saved: {stats['saved']} | "
        f"Corrupted removed: {stats['corrupted']} | "
        f"Duplicates removed: {stats['duplicates']}"
    )
    for cls_name in TARGET_CLASSES:
        count = len(list((OUTPUT_DIR / cls_name).glob("*.jpg")))
        print(f"  - {cls_name}: {count} images")


if __name__ == "__main__":
    curate_and_clean()
