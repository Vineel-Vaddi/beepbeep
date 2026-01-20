# utils/mongo_backend.py
import os
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import gridfs
import streamlit as st
from bson import ObjectId
from PIL import Image
from pymongo import MongoClient, ASCENDING

DB_NAME_DEFAULT = "beepbeep_DB"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------
# Secrets / Env helpers (FIXES your NameError + supports flat secrets)
# -----------------------------
def _get_mongo_uri() -> Optional[str]:
    """
    Supports:
      1) Streamlit Cloud flat secrets:
         MONGO_URI = "..."
      2) Nested secrets:
         [mongo]
         uri = "..."
      3) Env vars:
         MONGO_URI or MONGODB_URI
    """
    # flat secrets (your current format)
    uri = st.secrets.get("MONGO_URI", None)
    if uri:
        return str(uri).strip()

    # nested secrets
    mongo_block = st.secrets.get("mongo", None)
    if isinstance(mongo_block, dict):
        uri2 = mongo_block.get("uri") or mongo_block.get("MONGO_URI")
        if uri2:
            return str(uri2).strip()

    # env fallback
    env_uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
    if env_uri:
        return str(env_uri).strip()

    return None


def _get_db_name(default: str = DB_NAME_DEFAULT) -> str:
    """
    Supports:
      1) Streamlit Cloud flat secrets:
         DB_NAME = "..."
      2) Nested secrets:
         [mongo]
         db_name = "..."
      3) Env:
         DB_NAME
    """
    dbn = st.secrets.get("DB_NAME", None)
    if dbn:
        return str(dbn).strip()

    mongo_block = st.secrets.get("mongo", None)
    if isinstance(mongo_block, dict):
        dbn2 = mongo_block.get("db_name") or mongo_block.get("DB_NAME")
        if dbn2:
            return str(dbn2).strip()

    env_db = os.environ.get("DB_NAME")
    if env_db:
        return str(env_db).strip()

    return default


# -----------------------------
# DB / GridFS
# -----------------------------
@st.cache_resource(show_spinner=False)
def _get_client() -> MongoClient:
    uri = _get_mongo_uri()
    if not uri:
        raise RuntimeError(
            "Mongo URI missing. Set Streamlit secrets MONGO_URI (recommended) or env MONGO_URI."
        )
    return MongoClient(uri)


def get_db(db_name: Optional[str] = None):
    """
    Returns a connected DB handle.
    If db_name not provided, resolves from secrets/env/DB_NAME_DEFAULT.
    """
    client = _get_client()
    final_db = db_name or _get_db_name()
    return client[final_db]


def get_fs(db):
    # Keep your existing collection name for GridFS
    return gridfs.GridFS(db, collection="frames_fs")


def ensure_indexes(db):
    # project metadata
    db.videos.create_index([("video_id", ASCENDING)], unique=True)

    # clips
    db.clips.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)

    # annotations (one doc per clip)
    db.annotations.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)

    # app state (cursor, counts, filter)
    db.state.create_index([("video_id", ASCENDING)], unique=True)

    # GridFS file metadata (your ingest uses these)
    db.frames_fs.files.create_index(
        [("metadata.video_id", ASCENDING), ("metadata.filename", ASCENDING)],
        unique=True,
    )

    # Optional (recommended if you export YOLO-per-frame docs later)
    # db.yolo_labels.create_index([("video_id", ASCENDING), ("frame_file", ASCENDING)], unique=True)


# -----------------------------
# Project selector helpers
# -----------------------------
def list_videos(db) -> List[str]:
    return [d["video_id"] for d in db.videos.find({}, {"video_id": 1}).sort("video_id", 1)]


def load_state(db, video_id: str) -> Dict[str, Any]:
    return db.state.find_one({"video_id": video_id}) or {}


def save_state(db, video_id: str, state: Dict[str, Any]):
    db.state.update_one({"video_id": video_id}, {"$set": state}, upsert=True)


def get_video_stats(db, video_id: str) -> Dict[str, Any]:
    vid = db.videos.find_one({"video_id": video_id}) or {}
    st_doc = db.state.find_one({"video_id": video_id}) or {}

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


# -----------------------------
# Clips + annotations
# -----------------------------
def load_clips(db, video_id: str) -> List[Dict[str, Any]]:
    # Each clip doc: {video_id, clip_id, frame_files, frames([gridfs ids])}
    return list(db.clips.find({"video_id": video_id}, {"_id": 0}).sort("clip_id", 1))


def load_annotations_map(db, video_id: str) -> Dict[str, Dict[str, Any]]:
    # one doc per clip_id
    m: Dict[str, Dict[str, Any]] = {}
    for d in db.annotations.find({"video_id": video_id}, {"_id": 0}):
        m[d["clip_id"]] = d
    return m


def upsert_annotation(db, rec: Dict[str, Any]):
    db.annotations.update_one(
        {"video_id": rec["video_id"], "clip_id": rec["clip_id"]},
        {"$set": rec},
        upsert=True,
    )


def compute_progress_counts(
    clips: List[Dict[str, Any]],
    ann_map: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """
    Keeps your original progress logic:
    review_status ∈ {done, skipped, unclear, unlabeled}
    """
    counts = {"done": 0, "skipped": 0, "unclear": 0, "unlabeled": 0}
    for c in clips:
        cid = c["clip_id"]
        rec = ann_map.get(cid)
        status = rec.get("review_status", "unlabeled") if rec else "unlabeled"
        if status not in counts:
            status = "unlabeled"
        counts[status] += 1
    return counts


def get_last_labeled_clip_idx(
    clips: List[Dict[str, Any]],
    ann_map: Dict[str, Dict[str, Any]],
) -> Optional[int]:
    last = None
    for i, c in enumerate(clips):
        rec = ann_map.get(c["clip_id"])
        if rec and rec.get("review_status") in ("done", "skipped", "unclear"):
            last = i
    return last


# -----------------------------
# Frame loading (GridFS -> PIL)
# -----------------------------
def _to_objectid(x: Any) -> ObjectId:
    if isinstance(x, ObjectId):
        return x
    try:
        return ObjectId(str(x))
    except Exception as e:
        raise ValueError(f"Invalid GridFS frame_id: {x}") from e


@st.cache_data(show_spinner=False)
def _load_frame_bytes(frame_id_str: str) -> bytes:
    """
    Cached by frame_id string.
    NOTE: This opens DB inside cached function (OK for Streamlit),
    and uses cached client via _get_client().
    """
    db = get_db()
    fs = get_fs(db)
    oid = _to_objectid(frame_id_str)
    f = fs.get(oid)
    return f.read()


def load_frame_image_cached(video_id: str, frame_id: Any, max_w: int = 1100) -> Image.Image:
    """
    video_id is unused here but kept for your app signature compatibility.
    frame_id can be ObjectId or str(ObjectId).
    """
    raw = _load_frame_bytes(str(frame_id))
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if max_w and img.size[0] > max_w:
        w, h = img.size
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h))

    return img
