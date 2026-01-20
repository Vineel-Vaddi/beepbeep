import os
import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient, ASCENDING
import gridfs
from PIL import Image

DB_NAME_DEFAULT = "beepbeep_DB"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_db(db_name: str = DB_NAME_DEFAULT):
    uri = _get_mongo_uri_from_streamlit_secrets()
    if not uri:
        raise RuntimeError("Mongo URI missing. Set st.secrets[mongo][uri] or env MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    return db


def get_fs(db):
    return gridfs.GridFS(db, collection="frames_fs")


def ensure_indexes(db):
    db.videos.create_index([("video_id", ASCENDING)], unique=True)
    db.clips.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)
    db.annotations.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)
    db.state.create_index([("video_id", ASCENDING)], unique=True)
    db.frames_fs.files.create_index([("metadata.video_id", ASCENDING), ("metadata.filename", ASCENDING)], unique=True)


# -----------------------------
# Project selector helpers
# -----------------------------
def list_videos(db) -> List[str]:
    return [d["video_id"] for d in db.videos.find({}, {"video_id": 1}).sort("video_id", 1)]


def get_video_stats(db, video_id: str) -> Dict[str, Any]:
    vid = db.videos.find_one({"video_id": video_id}) or {}
    st_doc = db.state.find_one({"video_id": video_id}) or {}

    # Load minimal clip list + annotations and recompute counts
    clips = list(db.clips.find({"video_id": video_id}, {"clip_id": 1, "_id": 0}).sort("clip_id", 1))
    ann_map = load_annotations_map(db, video_id)
    counts = compute_progress_counts(clips, ann_map)

    last_cursor = st_doc.get("cursor_clip_idx", 0)
    total_clips = len(clips)

    return {
        "video_id": video_id,
        "total_frames": vid.get("kept_frames", vid.get("total_frames", 0)),
        "total_clips": total_clips,
        "done": counts.get("done", 0),
        "skipped": counts.get("skipped", 0),
        "unclear": counts.get("unclear", 0),
        "unlabeled": counts.get("unlabeled", 0),
        "cursor_clip_idx": last_cursor,
    }



def load_state(db, video_id: str) -> Dict[str, Any]:
    return db.state.find_one({"video_id": video_id}) or {}


def save_state(db, video_id: str, state: Dict[str, Any]):
    db.state.update_one({"video_id": video_id}, {"$set": state}, upsert=True)


# -----------------------------
# Clips + annotations
# -----------------------------
def load_clips(db, video_id: str) -> List[Dict[str, Any]]:
    # Each clip doc: {video_id, clip_id, frame_files, frames(gridfs ids)}
    return list(db.clips.find({"video_id": video_id}, {"_id": 0}).sort("clip_id", 1))


def load_annotations_map(db, video_id: str) -> Dict[str, Dict[str, Any]]:
    # latest doc per clip (we store single doc per clip_id anyway)
    m = {}
    for d in db.annotations.find({"video_id": video_id}, {"_id": 0}):
        m[d["clip_id"]] = d
    return m


def upsert_annotation(db, rec: Dict[str, Any]):
    db.annotations.update_one(
        {"video_id": rec["video_id"], "clip_id": rec["clip_id"]},
        {"$set": rec},
        upsert=True,
    )


def compute_progress_counts(clips: List[Dict[str, Any]], ann_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"done": 0, "skipped": 0, "unclear": 0, "unlabeled": 0}
    for c in clips:
        cid = c["clip_id"]
        rec = ann_map.get(cid)
        status = rec.get("review_status", "unlabeled") if rec else "unlabeled"
        if status not in counts:
            status = "unlabeled"
        counts[status] += 1
    return counts


def get_last_labeled_clip_idx(clips: List[Dict[str, Any]], ann_map: Dict[str, Dict[str, Any]]) -> Optional[int]:
    last = None
    for i, c in enumerate(clips):
        rec = ann_map.get(c["clip_id"])
        if rec and rec.get("review_status") in ("done", "skipped", "unclear"):
            last = i
    return last


# -----------------------------
# Frame loading (GridFS -> PIL)
# -----------------------------
def load_frame_image_cached(video_id: str, frame_id: Any, max_w: int = 1100) -> Image.Image:
    """
    frame_id can be:
    - ObjectId (ideal)
    - str(ObjectId) (current ingestion)
    We convert safely before GridFS.get().
    """
    import streamlit as st
    from bson import ObjectId

    def _to_objectid(x: Any) -> ObjectId:
        if isinstance(x, ObjectId):
            return x
        # common case: stored as "65f0...." string
        try:
            return ObjectId(str(x))
        except Exception as e:
            raise ValueError(f"Invalid GridFS frame_id: {x}") from e

    @st.cache_data(show_spinner=False)
    def _load_bytes(frame_id_str: str) -> bytes:
        db = get_db()
        fs = get_fs(db)
        oid = _to_objectid(frame_id_str)
        f = fs.get(oid)
        return f.read()

    raw = _load_bytes(str(frame_id))
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if max_w and img.size[0] > max_w:
        w, h = img.size
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h))

    return img

