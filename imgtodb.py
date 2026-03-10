"""
Image Folder / Kaggle -> MongoDB uploader

Uploads images into:
- MongoDB metadata collection: images
- MongoDB GridFS bucket: image_files

DB:
  traffic_violation

Main collection schema:
{
  "image_id": "unique_string",
  "dataset": "dataset_name",
  "filename": "image.jpg",
  "gridfs_id": ObjectId,
  "status": "unlabeled",
  "created_at": ISODate,
  "updated_at": ISODate,

  // extended fields for app use
  "width": 1920,
  "height": 1080,
  "mime_type": "image/jpeg",
  "file_size": 245123,
  "source": "local" | "kaggle"
}

Requirements:
  pip install -U pymongo kaggle python-dotenv pillow

Run:
  python upload_images_to_mongodb.py
"""

import os
import sys
import uuid
import zipfile
import shutil
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from gridfs import GridFS
from bson import ObjectId

load_dotenv()

DB_NAME = os.environ.get("DB_NAME", "traffic_violation")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "images")
GRIDFS_BUCKET = os.environ.get("GRIDFS_BUCKET", "image_files")

WORKDIR = Path("kaggle_downloads")
EXTRACT_DIR = WORKDIR / "extracted"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_IF_EXISTS = True


def utc_now():
    return datetime.now(timezone.utc)


def ensure_kaggle_ok():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        print("❌ Kaggle package not found. Install with: pip install kaggle")
        raise

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:
        print("\n❌ Kaggle authentication failed.")
        print("Fix:")
        print("  Option A: Put kaggle.json in ~/.kaggle/kaggle.json (or Windows equivalent)")
        print("  Option B: Set env vars KAGGLE_USERNAME and KAGGLE_KEY\n")
        raise
    return api


def download_kaggle_dataset(api, dataset: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading Kaggle dataset: {dataset}")
    api.dataset_download_files(dataset, path=str(out_dir), quiet=False, unzip=False)

    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip found after download in: {out_dir}")
    print(f"✅ Downloaded: {zips[0]}")
    return zips[0]


def unzip(zip_path: Path, extract_to: Path):
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("✅ Extracted")


def find_images(root: Path) -> List[Path]:
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images


def make_unique_image_id(dataset_name: str, filename: str) -> str:
    return f"{dataset_name}::{uuid.uuid4().hex}::{filename}"


def get_image_meta(img_path: Path) -> Tuple[int, int, str, int]:
    with Image.open(img_path) as img:
        width, height = img.size
        mime_type = Image.MIME.get(img.format)
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(str(img_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    file_size = img_path.stat().st_size
    return width, height, mime_type, file_size


def ensure_indexes(col):
    # Prevent duplicate file metadata within the same dataset
    col.create_index([("dataset", ASCENDING), ("filename", ASCENDING)], unique=True)
    col.create_index([("image_id", ASCENDING)], unique=True)
    col.create_index([("status", ASCENDING)])
    col.create_index([("gridfs_id", ASCENDING)], unique=True)


def get_source_input():
    print("\nSelect upload source:")
    print("1. Local folder")
    print("2. Kaggle dataset")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        local_path = input("Enter local folder path: ").strip()
        folder = Path(local_path)
        if not folder.exists() or not folder.is_dir():
            print(f"❌ Invalid folder: {local_path}")
            sys.exit(1)

        dataset_name = input(f"Enter Dataset_name_In_DB (press Enter to use '{folder.name}'): ").strip()
        if not dataset_name:
            dataset_name = folder.name

        return {
            "source_type": "local",
            "dataset_name": dataset_name,
            "images_root": folder,
            "kaggle_dataset": None,
        }

    if choice == "2":
        kaggle_dataset = input(
            "Enter Kaggle dataset (e.g., tkm22092/indian-number-plate-images): "
        ).strip()
        dataset_name = input("Enter Dataset_name_In_DB: ").strip()
        if not dataset_name:
            print("❌ Dataset_name_In_DB cannot be empty.")
            sys.exit(1)

        api = ensure_kaggle_ok()
        WORKDIR.mkdir(parents=True, exist_ok=True)

        try:
            zip_path = download_kaggle_dataset(api, kaggle_dataset, WORKDIR)
            unzip(zip_path, EXTRACT_DIR)
        except Exception as e:
            print(f"❌ Error downloading/extracting Kaggle dataset: {e}")
            sys.exit(1)

        return {
            "source_type": "kaggle",
            "dataset_name": dataset_name,
            "images_root": EXTRACT_DIR,
            "kaggle_dataset": kaggle_dataset,
        }

    print("❌ Invalid choice.")
    sys.exit(1)


def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        print("❌ Missing MONGODB_URI in environment.")
        sys.exit(1)

    config = get_source_input()
    source_type = config["source_type"]
    dataset_name = config["dataset_name"]
    images_root = config["images_root"]
    kaggle_dataset = config["kaggle_dataset"]

    images = find_images(images_root)
    if not images:
        print(f"❌ No images found in: {images_root}")
        sys.exit(1)

    print(f"🖼️ Found {len(images)} images.")

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    fs = GridFS(db, collection=GRIDFS_BUCKET)

    try:
        ensure_indexes(col)
    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")

    uploaded = 0
    skipped = 0
    failed = 0

    for idx, img_path in enumerate(images, start=1):
        filename = img_path.name

        if SKIP_IF_EXISTS:
            existing = col.find_one({"dataset": dataset_name, "filename": filename}, {"_id": 1})
            if existing:
                skipped += 1
                continue

        gridfs_id: ObjectId | None = None
        try:
            width, height, mime_type, file_size = get_image_meta(img_path)

            with open(img_path, "rb") as f:
                gridfs_id = fs.put(
                    f,
                    filename=filename,
                    contentType=mime_type,
                    metadata={
                        "dataset": dataset_name,
                        "source": source_type,
                        "kaggle_dataset": kaggle_dataset,
                        "original_path": str(img_path),
                        "width": width,
                        "height": height,
                        "file_size": file_size,
                    },
                )

            now = utc_now()
            doc = {
                "image_id": make_unique_image_id(dataset_name, filename),
                "dataset": dataset_name,
                "filename": filename,
                "gridfs_id": gridfs_id,
                "status": "unlabeled",
                "created_at": now,
                "updated_at": now,
                "width": width,
                "height": height,
                "mime_type": mime_type,
                "file_size": file_size,
                "source": source_type,
            }

            if kaggle_dataset:
                doc["kaggle_dataset"] = kaggle_dataset

            col.insert_one(doc)
            uploaded += 1

            if uploaded % 100 == 0 or idx == len(images):
                print(f"✅ Progress: uploaded={uploaded}, skipped={skipped}, failed={failed}")

        except DuplicateKeyError:
            skipped += 1
            if gridfs_id is not None:
                try:
                    fs.delete(gridfs_id)
                except Exception:
                    pass
        except Exception as e:
            failed += 1
            print(f"❌ Failed: {img_path} | {e}")
            if gridfs_id is not None:
                try:
                    fs.delete(gridfs_id)
                except Exception:
                    pass

    print("\n====================")
    print("DONE")
    print("====================")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"DB      : {DB_NAME}")
    print(f"COLL    : {COLLECTION_NAME}")
    print(f"GRIDFS  : {GRIDFS_BUCKET}")
    print(f"Dataset : {dataset_name}")


if __name__ == "__main__":
    main()