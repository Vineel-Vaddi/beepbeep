import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


import io
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from PIL import Image

from utils.frames import dhash, hamming
from utils.mongo_store import (
    get_client,
    get_db,
    get_fs,
    ensure_indexes,
    put_frame_bytes,
    find_frame_file,
)

DB_NAME = os.environ.get("DB_NAME", "beepbeep_DB")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sorted_frame_files(frames_dir: str) -> List[str]:
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files


def ensure_ingest_log_dir() -> str:
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def log_path_for(video_id: str) -> str:
    log_dir = ensure_ingest_log_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"ingest_{video_id}_{ts}.json")


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dedupe_frames(frames_dir: str, dhash_threshold: int = 6) -> Dict[str, Any]:
    files = sorted_frame_files(frames_dir)
    kept, dropped = [], []
    last_hash: Optional[int] = None

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
        group = kept_frames[i : i + clip_len]
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
) -> None:
    client = get_client(mongo_uri)
    db = get_db(client, DB_NAME)
    fs = get_fs(db)
    ensure_indexes(db)

    frames_dir = os.path.join(dataset_root, video_id)
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")

    # ---- Ingest log (JSON) ----
    ingest_log: Dict[str, Any] = {
        "run_at": now_iso(),
        "video_id": video_id,
        "dataset_root": dataset_root,
        "frames_dir": frames_dir,
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

    # 0) Optional: if video already ingested and you don't want rebuild
    # (You can comment this out if you always want to rebuild clips.)
    existing_video = db.videos.find_one({"video_id": video_id}, {"_id": 1, "updated_at": 1})
    if existing_video and not overwrite_clips:
        ingest_log["notes"].append(
            "Video already exists in db.videos. Frames upload will skip duplicates; clips will be upserted (no overwrite)."
        )

    # 1) dedupe locally (near-duplicate filter)
    dd = dedupe_frames(frames_dir, dhash_threshold=dhash_threshold)
    kept, dropped = dd["kept_frames"], dd["dropped_frames"]
    ingest_log["frames"]["kept"] = kept
    ingest_log["frames"]["dropped_by_dedupe"] = dropped

    # 2) upload frames (kept only) with strong duplicate prevention
    #    - First: check GridFS by metadata (your existing find_frame_file)
    #    - Second: ensure we never upload same filename twice within this run
    frame_ids: Dict[str, Any] = {}
    seen_names = set()

    for fname in kept:
        if fname in seen_names:
            ingest_log["frames"]["already_in_mongo"].append(fname)
            continue
        seen_names.add(fname)

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

    # 3) build clips + store
    clips = build_clips(kept, clip_len=clip_len)

    if overwrite_clips:
        db.clips.delete_many({"video_id": video_id})

    # IMPORTANT: clips refer to filenames via frame_files, and ids via frames
    clip_docs = []
    for c in clips:
        # If any frame id is missing (upload failed), skip clip to prevent broken clip docs
        missing = [f for f in c["frames"] if f not in frame_ids]
        if missing:
            ingest_log["notes"].append(
                f"Skipped clip_id={c['clip_id']} due to missing frame_ids for: {missing}"
            )
            continue

        clip_docs.append(
            {
                "video_id": video_id,
                "clip_id": c["clip_id"],
                "frame_files": c["frames"],  # filenames
                "frames": [frame_ids[f] for f in c["frames"]],  # GridFS ids
                "created_at": now_iso(),
            }
        )

    # upsert clips
    # - If clip doc exists, keep it unchanged unless overwrite_clips=True (already deleted then)
    for doc in clip_docs:
        db.clips.update_one(
            {"video_id": doc["video_id"], "clip_id": doc["clip_id"]},
            {"$setOnInsert": doc},
            upsert=True,
        )

    # 4) upsert video summary
    total_frames_local = len(sorted_frame_files(frames_dir))
    db.videos.update_one(
        {"video_id": video_id},
        {
            "$set": {
                "video_id": video_id,
                "dataset_root": dataset_root,
                "total_frames": total_frames_local,
                "kept_frames": len(kept),
                "dropped_frames": len(dropped),
                "dhash_threshold": dhash_threshold,
                "clip_len": clip_len,
                "updated_at": now_iso(),
            }
        },
        upsert=True,
    )

    # 5) ensure state exists (don’t reset if already exists!)
    #    Only set on insert. This preserves resume/progress.
    db.state.update_one(
        {"video_id": video_id},
        {
            "$setOnInsert": {
                "video_id": video_id,
                "cursor_clip_idx": 0,
                "filter_mode": "All",
                "counts": {"done": 0, "skipped": 0, "unclear": 0, "unlabeled": len(clip_docs)},
                "updated_at": now_iso(),
            }
        },
        upsert=True,
    )

    # ---- finalize ingest log ----
    ingest_log["counts"] = {
        "total_frames_local": total_frames_local,
        "kept_after_dedupe": len(kept),
        "dropped_by_dedupe": len(dropped),
        "already_in_mongo": len(ingest_log["frames"]["already_in_mongo"]),
        "uploaded_now": len(ingest_log["frames"]["uploaded_now"]),
        "failed_uploads": len(ingest_log["frames"]["failed"]),
        "clips_built": len(clips),
        "clips_written": len(clip_docs),
    }

    log_file = log_path_for(video_id)
    write_json(log_file, ingest_log)

    print(f"✅ Ingested video_id={video_id}")
    print(
        f"   total_frames={total_frames_local} kept={len(kept)} dropped={len(dropped)} "
        f"uploaded_now={len(ingest_log['frames']['uploaded_now'])} already_in_mongo={len(ingest_log['frames']['already_in_mongo'])} "
        f"clips={len(clip_docs)}"
    )
    print(f"🧾 Log written: {log_file}")


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
