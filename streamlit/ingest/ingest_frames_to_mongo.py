import os
import io
from datetime import datetime
from typing import List, Dict, Any

from PIL import Image

from utils.frames import dhash, hamming
from utils.mongo_store import get_client, get_db, get_fs, ensure_indexes, put_frame_bytes, find_frame_file

DB_NAME = "beepbeep_DB"

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def sorted_frame_files(frames_dir: str) -> List[str]:
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    files.sort()
    return files

def dedupe_frames(frames_dir: str, dhash_threshold: int = 6) -> Dict[str, Any]:
    files = sorted_frame_files(frames_dir)
    kept, dropped = [], []
    last_hash = None

    for f in files:
        p = os.path.join(frames_dir, f)
        try:
            img = Image.open(p).convert("RGB")
            hv = dhash(img)
        except Exception:
            dropped.append(f)
            continue

        if last_hash is None:
            kept.append(f)
            last_hash = hv
        else:
            if hamming(last_hash, hv) >= dhash_threshold:
                kept.append(f)
                last_hash = hv
            else:
                dropped.append(f)

    return {"kept_frames": kept, "dropped_frames": dropped, "dhash_threshold": dhash_threshold}

def build_clips(kept_frames: List[str], clip_len: int = 4) -> List[Dict[str, Any]]:
    clip_len = max(3, min(6, int(clip_len)))
    clips = []
    clip_idx = 0
    for i in range(0, len(kept_frames), clip_len):
        group = kept_frames[i:i+clip_len]
        if not group:
            continue
        clips.append({"clip_id": f"{clip_idx:06d}", "frames": group})
        clip_idx += 1
    return clips

def image_to_jpg_bytes(path: str, quality: int = 90) -> bytes:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def ingest_video(
    mongo_uri: str,
    dataset_root: str,
    video_id: str,
    dhash_threshold: int = 6,
    clip_len: int = 4,
    jpg_quality: int = 90,
    overwrite_clips: bool = False,
):
    client = get_client(mongo_uri)
    db = get_db(client, DB_NAME)
    fs = get_fs(db)
    ensure_indexes(db)

    frames_dir = os.path.join(dataset_root, video_id)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    # 1) dedupe
    dd = dedupe_frames(frames_dir, dhash_threshold=dhash_threshold)
    kept, dropped = dd["kept_frames"], dd["dropped_frames"]

    # 2) upload frames (kept only)
    frame_ids = {}
    for fname in kept:
        existing = find_frame_file(fs, db, video_id, fname)
        if existing:
            frame_ids[fname] = existing
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

    # 3) build clips + store
    clips = build_clips(kept, clip_len=clip_len)

    if overwrite_clips:
        db.clips.delete_many({"video_id": video_id})

    clip_docs = []
    for c in clips:
        clip_docs.append({
            "video_id": video_id,
            "clip_id": c["clip_id"],
            "frame_files": c["frames"],
            "frames": [frame_ids[f] for f in c["frames"]],
            "created_at": now_iso(),
        })

    # upsert clips
    for doc in clip_docs:
        db.clips.update_one(
            {"video_id": doc["video_id"], "clip_id": doc["clip_id"]},
            {"$setOnInsert": doc},
            upsert=True
        )

    # 4) upsert video summary
    db.videos.update_one(
        {"video_id": video_id},
        {"$set": {
            "video_id": video_id,
            "dataset_root": dataset_root,
            "total_frames": len(sorted_frame_files(frames_dir)),
            "kept_frames": len(kept),
            "dropped_frames": len(dropped),
            "dhash_threshold": dhash_threshold,
            "clip_len": clip_len,
            "updated_at": now_iso(),
        }},
        upsert=True
    )

    # 5) ensure state exists
    db.state.update_one(
        {"video_id": video_id},
        {"$setOnInsert": {
            "video_id": video_id,
            "cursor_clip_idx": 0,
            "filter_mode": "All",
            "counts": {"done": 0, "skipped": 0, "unclear": 0, "unlabeled": len(clips)},
            "updated_at": now_iso(),
        }},
        upsert=True
    )

    print(f"✅ Ingested video_id={video_id}")
    print(f"   total_frames={len(sorted_frame_files(frames_dir))} kept={len(kept)} dropped={len(dropped)} clips={len(clips)}")

if __name__ == "__main__":
    # DO NOT hardcode credentials in production; use env var
    mongo_uri = os.environ.get("MONGO_URI", "").strip()
    if not mongo_uri:
        raise RuntimeError("Set MONGO_URI env var")

    dataset_root = os.environ.get("DATASET_ROOT", "frames_out")
    video_id = os.environ.get("VIDEO_ID")
    if not video_id:
        raise RuntimeError("Set VIDEO_ID env var")

    ingest_video(
        mongo_uri=mongo_uri,
        dataset_root=dataset_root,
        video_id=video_id,
        dhash_threshold=int(os.environ.get("DHASH_THRESHOLD", "6")),
        clip_len=int(os.environ.get("CLIP_LEN", "4")),
        jpg_quality=int(os.environ.get("JPG_QUALITY", "90")),
        overwrite_clips=os.environ.get("OVERWRITE_CLIPS", "0") == "1",
    )
