import os
import sys
import io
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from PIL import Image

# ------------------------------------------------------------
# PATH SETUP
# Repo layout (your case):
#   beepbeep/
#     streamlit/
#       ingest/ingest_frames_to_mongo.py   <-- this file
#       utils/mongo_store.py
#     yt_to_frames/frames_out/<video_id>/frame_*.jpg
#
# So PROJECT_ROOT = beepbeep/
# STREAMLIT_ROOT = beepbeep/streamlit
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
STREAMLIT_ROOT = os.path.join(PROJECT_ROOT, "beepbeep_app")

# Make "utils.*" importable (from streamlit/utils)
if STREAMLIT_ROOT not in sys.path:
    sys.path.insert(0, STREAMLIT_ROOT)

# ------------------------------------------------------------
# Load .env (prefer streamlit/.env, else repo-root .env)
# ------------------------------------------------------------
ENV_CANDIDATES = [
    os.path.join(STREAMLIT_ROOT, ".env"),   # most likely in your setup
    os.path.join(PROJECT_ROOT, ".env"),
]
ENV_PATH_USED = None
for p in ENV_CANDIDATES:
    if os.path.isfile(p):
        load_dotenv(dotenv_path=p)
        ENV_PATH_USED = p
        break

# ------------------------------------------------------------
# Mongo helpers (your file)
# ------------------------------------------------------------
from utils.mongo_store import (
    get_client,
    get_db,
    get_fs,
    ensure_indexes,
    put_frame_bytes,
    find_frame_file,
)

DB_NAME = os.environ.get("DB_NAME", "beepbeep_DB").strip()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_ingest_log_dir() -> str:
    log_dir = os.path.join(THIS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_path_for(video_id: str) -> str:
    log_dir = ensure_ingest_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_vid = video_id.replace("\\", "_").replace("/", "_")
    return os.path.join(log_dir, f"ingest_{safe_vid}_{ts}.json")


def sorted_frame_files(frames_dir: str) -> List[str]:
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files


def has_images(frames_dir: str) -> bool:
    try:
        for f in os.listdir(frames_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
    except Exception:
        return False
    return False


def list_video_folders(dataset_root: str) -> List[str]:
    dataset_root = os.path.abspath(dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"DATASET_ROOT not found: {dataset_root}")

    items = []
    for name in os.listdir(dataset_root):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p):
            items.append(name)
    items.sort()
    return items


def image_to_jpg_bytes(path: str, quality: int = 90) -> bytes:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------
# dHash + Hamming (NO streamlit dependency)
# ------------------------------------------------------------
def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Difference hash (dHash). Returns integer bit-hash.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())

    diff_bits = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            diff_bits.append(left > right)

    v = 0
    for bit in diff_bits:
        v = (v << 1) | int(bit)
    return v


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_frames(frames_dir: str, dhash_threshold: int = 6) -> Dict[str, Any]:
    """
    Sequential near-duplicate removal:
    - keep first
    - keep a frame only if hamming(dhash(current), dhash(last_kept)) >= threshold
    """
    files = sorted_frame_files(frames_dir)
    kept, dropped = [], []
    last_hash: Optional[int] = None

    for fname in files:
        path = os.path.join(frames_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            hv = dhash(img)
        except Exception:
            dropped.append(fname)
            continue

        if last_hash is None:
            kept.append(fname)
            last_hash = hv
        else:
            if hamming(last_hash, hv) >= dhash_threshold:
                kept.append(fname)
                last_hash = hv
            else:
                dropped.append(fname)

    return {"kept_frames": kept, "dropped_frames": dropped, "dhash_threshold": dhash_threshold}


def build_clips(kept_frames: List[str], clip_len: int = 4) -> List[Dict[str, Any]]:
    """
    Non-overlapping grouping. clip_len forced to [3..6].
    clip_id: 000000, 000001, ...
    """
    clip_len = max(3, min(6, int(clip_len)))
    clips = []
    clip_idx = 0

    for i in range(0, len(kept_frames), clip_len):
        group = kept_frames[i : i + clip_len]
        if not group:
            continue
        clips.append({"clip_id": f"{clip_idx:06d}", "frames": group})
        clip_idx += 1

    return clips


# ------------------------------------------------------------
# Ingest one video folder
# ------------------------------------------------------------
def ingest_video(
    db,
    fs,
    dataset_root: str,
    video_id: str,
    dhash_threshold: int = 6,
    clip_len: int = 4,
    jpg_quality: int = 90,
    overwrite_clips: bool = False,
) -> None:
    ensure_indexes(db)

    dataset_root = os.path.abspath(dataset_root)
    frames_dir = os.path.join(dataset_root, video_id)

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    if not has_images(frames_dir):
        print(f"⚠️ Skipping {video_id} (no image files found)")
        return

    ingest_log: Dict[str, Any] = {
        "run_at": now_iso(),
        "video_id": video_id,
        "dataset_root": dataset_root,
        "frames_dir": frames_dir,
        "project_root": PROJECT_ROOT,
        "streamlit_root": STREAMLIT_ROOT,
        "env_path_used": ENV_PATH_USED,
        "params": {
            "dhash_threshold": dhash_threshold,
            "clip_len": clip_len,
            "jpg_quality": jpg_quality,
            "overwrite_clips": overwrite_clips,
        },
        "counts": {},
        "frames": {
            "kept": [],
            "dropped_by_dedupe": [],
            "already_in_mongo": [],
            "uploaded_now": [],
            "failed": [],
        },
        "notes": [],
    }

    existing_video = db.videos.find_one({"video_id": video_id}, {"_id": 1, "updated_at": 1})

    # 1) dedupe locally
    dd = dedupe_frames(frames_dir, dhash_threshold=dhash_threshold)
    kept, dropped = dd["kept_frames"], dd["dropped_frames"]
    ingest_log["frames"]["kept"] = kept
    ingest_log["frames"]["dropped_by_dedupe"] = dropped

    # 2) upload kept frames (skip duplicates using GridFS metadata unique index)
    frame_ids: Dict[str, str] = {}
    for fname in kept:
        try:
            existing = find_frame_file(fs, db, video_id, fname)
            if existing:
                frame_ids[fname] = existing
                ingest_log["frames"]["already_in_mongo"].append(fname)
                continue

            path = os.path.join(frames_dir, fname)
            jpg_bytes = image_to_jpg_bytes(path, quality=jpg_quality)

            fid = put_frame_bytes(
                fs,
                jpg_bytes,
                filename=f"{video_id}/{fname}",
                metadata={"video_id": video_id, "filename": fname, "created_at": now_iso()},
            )
            frame_ids[fname] = fid
            ingest_log["frames"]["uploaded_now"].append(fname)

        except Exception as e:
            ingest_log["frames"]["failed"].append({"filename": fname, "error": str(e)})

    total_frames_local = len(sorted_frame_files(frames_dir))

    # IMPORTANT: keep clip grouping stable unless overwrite_clips=True
    if existing_video and not overwrite_clips:
        ingest_log["notes"].append(
            "Video exists and overwrite_clips=False -> skipping clip rebuild to keep clip grouping stable. "
            "Only syncing frames (uploading missing frames)."
        )

        db.videos.update_one(
            {"video_id": video_id},
            {"$set": {"video_id": video_id, "dataset_root": dataset_root, "total_frames": total_frames_local, "updated_at": now_iso()}},
            upsert=True,
        )

        clips_in_db = int(db.clips.count_documents({"video_id": video_id}))

        ingest_log["counts"] = {
            "total_frames_local": total_frames_local,
            "kept_after_dedupe": len(kept),
            "dropped_by_dedupe": len(dropped),
            "already_in_mongo": len(ingest_log["frames"]["already_in_mongo"]),
            "uploaded_now": len(ingest_log["frames"]["uploaded_now"]),
            "failed_uploads": len(ingest_log["frames"]["failed"]),
            "clips_built": 0,
            "clips_written": 0,
            "clips_in_db_after": clips_in_db,
        }

        log_file = log_path_for(video_id)
        write_json(log_file, ingest_log)

        print(f"✅ Video exists; frame sync done for video_id={video_id}")
        print(f"   total_frames={total_frames_local} uploaded_now={len(ingest_log['frames']['uploaded_now'])} already_in_mongo={len(ingest_log['frames']['already_in_mongo'])}")
        print(f"🧾 Log written: {log_file}")
        return

    # 3) build clips (new video OR overwrite)
    clips = build_clips(kept, clip_len=clip_len)

    if overwrite_clips:
        db.clips.delete_many({"video_id": video_id})
        ingest_log["notes"].append("overwrite_clips=True -> deleted existing clips for this video_id")

    clip_docs = []
    for c in clips:
        missing = [f for f in c["frames"] if f not in frame_ids]
        if missing:
            ingest_log["notes"].append(f"Skipped clip_id={c['clip_id']} due to missing frame_ids for: {missing}")
            continue

        clip_docs.append(
            {
                "video_id": video_id,
                "clip_id": c["clip_id"],
                "frame_files": c["frames"],                     # filenames
                "frames": [frame_ids[f] for f in c["frames"]],  # GridFS ids (strings)
                "created_at": now_iso(),
            }
        )

    for doc in clip_docs:
        db.clips.update_one(
            {"video_id": doc["video_id"], "clip_id": doc["clip_id"]},
            {"$setOnInsert": doc},
            upsert=True,
        )

    # 4) upsert video summary
    db.videos.update_one(
        {"video_id": video_id},
        {"$set": {
            "video_id": video_id,
            "dataset_root": dataset_root,
            "total_frames": total_frames_local,
            "kept_frames": len(kept),
            "dropped_frames": len(dropped),
            "dhash_threshold": dhash_threshold,
            "clip_len": clip_len,
            "updated_at": now_iso(),
        }},
        upsert=True,
    )

    total_clips_db = int(db.clips.count_documents({"video_id": video_id}))

    # 5) ensure state exists (don’t reset)
    db.state.update_one(
        {"video_id": video_id},
        {"$setOnInsert": {
            "video_id": video_id,
            "cursor_clip_idx": 0,
            "filter_mode": "All",
            "counts": {"done": 0, "skipped": 0, "unclear": 0, "unlabeled": total_clips_db},
            "updated_at": now_iso(),
        }},
        upsert=True,
    )

    ingest_log["counts"] = {
        "total_frames_local": total_frames_local,
        "kept_after_dedupe": len(kept),
        "dropped_by_dedupe": len(dropped),
        "already_in_mongo": len(ingest_log["frames"]["already_in_mongo"]),
        "uploaded_now": len(ingest_log["frames"]["uploaded_now"]),
        "failed_uploads": len(ingest_log["frames"]["failed"]),
        "clips_built": len(clips),
        "clips_written": len(clip_docs),
        "clips_in_db_after": total_clips_db,
    }

    log_file = log_path_for(video_id)
    write_json(log_file, ingest_log)

    print(f"✅ Ingested video_id={video_id}")
    print(f"   total_frames={total_frames_local} uploaded_now={len(ingest_log['frames']['uploaded_now'])} already_in_mongo={len(ingest_log['frames']['already_in_mongo'])} clips={total_clips_db}")
    print(f"🧾 Log written: {log_file}")


# ------------------------------------------------------------
# MAIN (batch or single)
# ------------------------------------------------------------
def resolve_dataset_root() -> str:
    """
    DATASET_ROOT rules:
    - If env provides absolute path -> use it
    - If env provides relative path -> resolve relative to PROJECT_ROOT
    - If env missing/empty -> default to PROJECT_ROOT/yt_to_frames/frames_out
    """
    ds = os.environ.get("DATASET_ROOT", "").strip()
    if not ds:
        ds = "yt_to_frames/frames_out"

    if os.path.isabs(ds):
        dataset_root = ds
    else:
        dataset_root = os.path.join(PROJECT_ROOT, ds)

    dataset_root = os.path.abspath(dataset_root)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            "DATASET_ROOT not found.\n"
            f"  Tried: {dataset_root}\n"
            "Fix: set DATASET_ROOT in .env to your absolute path, e.g.\n"
            r"  DATASET_ROOT=C:\Users\HP\Documents\GitHub\beepbeep\yt_to_frames\frames_out"
        )

    return dataset_root


if __name__ == "__main__":
    mongo_uri = os.environ.get("MONGO_URI", "").strip()
    if not mongo_uri:
        raise RuntimeError(
            "MONGO_URI not set. Put it in streamlit/.env or repo-root .env.\n"
            f"Checked: {ENV_CANDIDATES}"
        )

    dataset_root = resolve_dataset_root()

    only_video_id = os.environ.get("VIDEO_ID", "").strip()
    dhash_threshold = int(os.environ.get("DHASH_THRESHOLD", "6"))
    clip_len = int(os.environ.get("CLIP_LEN", "4"))
    jpg_quality = int(os.environ.get("JPG_QUALITY", "90"))
    overwrite_clips = os.environ.get("OVERWRITE_CLIPS", "0").strip() == "1"

    # Connect ONCE for batch speed
    client = get_client(mongo_uri)
    db = get_db(client, DB_NAME)
    fs = get_fs(db)
    ensure_indexes(db)

    # Batch log
    batch_log = {
        "run_at": now_iso(),
        "project_root": PROJECT_ROOT,
        "streamlit_root": STREAMLIT_ROOT,
        "env_path_used": ENV_PATH_USED,
        "dataset_root": dataset_root,
        "params": {
            "dhash_threshold": dhash_threshold,
            "clip_len": clip_len,
            "jpg_quality": jpg_quality,
            "overwrite_clips": overwrite_clips,
        },
        "videos_requested": only_video_id if only_video_id else "ALL",
        "videos_found": [],
        "videos_processed": [],
        "videos_skipped": [],
        "videos_failed": [],
    }

    video_ids = list_video_folders(dataset_root)
    batch_log["videos_found"] = video_ids

    if only_video_id:
        if only_video_id not in video_ids:
            raise RuntimeError(
                f"VIDEO_ID '{only_video_id}' not found under DATASET_ROOT '{dataset_root}'. "
                f"Examples: {video_ids[:10]}"
            )
        video_ids = [only_video_id]

    print("📦 Batch ingest starting")
    print(f"   PROJECT_ROOT={PROJECT_ROOT}")
    print(f"   ENV_PATH_USED={ENV_PATH_USED}")
    print(f"   DATASET_ROOT={dataset_root}")
    print(f"🎞️ Videos to ingest: {len(video_ids)}")

    for vid in video_ids:
        try:
            print(f"\n▶ Ingesting: {vid}")
            ingest_video(
                db=db,
                fs=fs,
                dataset_root=dataset_root,
                video_id=vid,
                dhash_threshold=dhash_threshold,
                clip_len=clip_len,
                jpg_quality=jpg_quality,
                overwrite_clips=overwrite_clips,
            )
            batch_log["videos_processed"].append(vid)
        except FileNotFoundError as e:
            batch_log["videos_skipped"].append({"video_id": vid, "reason": str(e)})
        except Exception as e:
            batch_log["videos_failed"].append({"video_id": vid, "error": str(e)})

    # Master batch log
    log_dir = ensure_ingest_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_path = os.path.join(log_dir, f"batch_ingest_{ts}.json")
    write_json(batch_log_path, batch_log)

    print("\n✅ Batch ingest completed")
    print(f"🧾 Master batch log written: {batch_log_path}")
    print(
        f"   processed={len(batch_log['videos_processed'])} "
        f"skipped={len(batch_log['videos_skipped'])} failed={len(batch_log['videos_failed'])}"
    )
