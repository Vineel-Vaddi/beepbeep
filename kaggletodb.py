"""
Kaggle -> MongoDB (GridFS) uploader

Uploads images from Kaggle dataset:
  tkm22092/indian-number-plate-images

MongoDB:
  DB: traffic_violation
  Collection: images
  Uses GridFS for actual image bytes.

Each image document schema:
{
  "image_id": "unique_string",
  "dataset": "dataset_name",
  "filename": "image.jpg",
  "gridfs_id": "ObjectId_of_file_in_GridFS",
  "status": "unlabeled",
  "created_at": "ISODate",
  "updated_at": "ISODate"
}

Requirements:
  pip install -U pymongo gridfs kaggle

Prereqs:
  1) Kaggle API credentials:
     - Place kaggle.json at:
         Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
         Linux/Mac: ~/.kaggle/kaggle.json
     OR set env vars KAGGLE_USERNAME and KAGGLE_KEY

  2) MongoDB URI in env var:
       MONGODB_URI="mongodb+srv://...."
     (recommended, do NOT hardcode in code)

Run:
  python kaggle_to_mongo_gridfs.py
"""

import os
import sys
import uuid
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from pymongo import MongoClient, ASCENDING
import gridfs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Default User Config Context
# -----------------------------
DB_NAME = os.environ.get("DB_NAME", "traffic_violation")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "images")

WORKDIR = Path("kaggle_downloads")  # local working folder
DOWNLOAD_ZIP = WORKDIR / "dataset.zip"
EXTRACT_DIR = WORKDIR / "extracted"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# If True, skip uploading if a doc with same dataset+filename exists
SKIP_IF_EXISTS = True


# -----------------------------
# Helpers
# -----------------------------
def utc_now():
    return datetime.now(timezone.utc)

def ensure_kaggle_ok():
    """
    Checks Kaggle auth quickly. If kaggle isn't installed/authenticated, exits with help.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("❌ Kaggle package not found. Install with: pip install kaggle")
        raise

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:
        print("\n❌ Kaggle authentication failed.")
        print("Fix:")
        print("  Option A: Put kaggle.json in ~/.kaggle/kaggle.json (or Windows equivalent)")
        print("  Option B: Set env vars KAGGLE_USERNAME and KAGGLE_KEY")
        print("Then rerun.\n")
        raise
    return api

def download_kaggle_dataset(api, dataset: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # kaggle API downloads as zip (datasetname.zip) into out_dir
    print(f"⬇️ Downloading Kaggle dataset: {dataset}")
    api.dataset_download_files(dataset, path=str(out_dir), quiet=False, unzip=False)

    # Find the zip
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
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    imgs.sort()
    return imgs

def make_unique_image_id(dataset_name: str, filename: str) -> str:
    # Stable unique ID: dataset + uuid4 (safe for multiple uploads)
    return f"{dataset_name}::{uuid.uuid4().hex}::{filename}"

def ensure_indexes(col):
    # Helps prevent duplicates and speeds checks
    col.create_index([("dataset", ASCENDING), ("filename", ASCENDING)], unique=True)
    col.create_index([("image_id", ASCENDING)], unique=True)


# -----------------------------
# Main Upload
# -----------------------------
def main():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        print("❌ Missing MONGODB_URI env var. Set it like:")
        print('   Windows (PowerShell): $env:MONGODB_URI="mongodb+srv://..."')
        print('   Linux/Mac: export MONGODB_URI="mongodb+srv://..."')
        sys.exit(1)

    print("\nSelect upload source:")
    print("1. Local folder")
    print("2. Kaggle dataset")
    choice = input("Enter choice (1 or 2): ").strip()
    
    source_type = ""
    kaggle_dataset_val = ""
    extract_dir = EXTRACT_DIR

    if choice == "1":
        source_type = "local"
        local_path = input("Enter local folder path: ").strip()
        extract_dir = Path(local_path)
        if not extract_dir.exists() or not extract_dir.is_dir():
             print(f"❌ Invalid or missing folder: {local_path}")
             sys.exit(1)
        dataset_name_in_db = extract_dir.name
        print(f"📁 Using local folder: {extract_dir}")
        print(f"📁 Auto-assigned Dataset_name_In_DB: {dataset_name_in_db}")
        
    elif choice == "2":
        source_type = "kaggle"
        kaggle_dataset_val = input("Enter Kaggle dataset (e.g., tkm22092/indian-number-plate-images): ").strip()
        dataset_name_in_db = input("Enter Dataset_name_In_DB: ").strip()
        
        # Kaggle download
        api = ensure_kaggle_ok()
        WORKDIR.mkdir(parents=True, exist_ok=True)
    
        try:
             zip_path = download_kaggle_dataset(api, kaggle_dataset_val, WORKDIR)
             unzip(zip_path, EXTRACT_DIR)
        except Exception as e:
             print(f"❌ Error downloading/extracting from Kaggle: {e}")
             sys.exit(1)
             
    else:
        print("❌ Invalid choice. Exiting.")
        sys.exit(1)

    if not dataset_name_in_db:
         print("❌ Dataset_name_In_DB cannot be empty.")
         sys.exit(1)

    # Gather images
    images = find_images(extract_dir)
    if not images:
        print(f"❌ No images found under: {extract_dir}")
        sys.exit(1)

    print(f"🖼️ Found {len(images)} images to upload.")

    # 3) Mongo + GridFS
    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    fs = gridfs.GridFS(db)

    # Ensure indexes (unique dataset+filename)
    try:
        ensure_indexes(col)
    except Exception as e:
        # If duplicates already exist, index creation can fail; user can resolve
        print(f"⚠️ Index creation warning: {e}")

    uploaded = 0
    skipped = 0
    failed = 0

    for i, img_path in enumerate(images, start=1):
        # Use only file name for schema (as requested)
        filename = img_path.name

        if SKIP_IF_EXISTS:
            existing = col.find_one({"dataset": dataset_name_in_db, "filename": filename}, {"_id": 1})
            if existing:
                skipped += 1
                if skipped % 200 == 0:
                    print(f"↩️ Skipped {skipped} (already exists)...")
                continue

        try:
            # Store binary in GridFS
            with open(img_path, "rb") as f:
                gridfs_id = fs.put(
                    f,
                    filename=filename,
                    dataset=dataset_name_in_db,
                    source=source_type,
                    kaggle_dataset=kaggle_dataset_val,
                )

            now = utc_now()
            doc = {
                "image_id": make_unique_image_id(dataset_name_in_db, filename),
                "dataset": dataset_name_in_db,
                "filename": filename,
                "gridfs_id": gridfs_id,
                "status": "unlabeled",
                "created_at": now,
                "updated_at": now,
            }

            col.insert_one(doc)
            uploaded += 1

            if uploaded % 200 == 0:
                print(f"✅ Uploaded {uploaded}/{len(images)}... (latest: {filename})")

        except Exception as e:
            failed += 1
            print(f"❌ Failed: {img_path} | {e}")

    print("\n====================")
    print("DONE")
    print("====================")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")
    print(f"DB      : {DB_NAME}")
    print(f"COLL    : {COLLECTION_NAME}")
    print(f"Dataset : {dataset_name_in_db}")

if __name__ == "__main__":
    main()