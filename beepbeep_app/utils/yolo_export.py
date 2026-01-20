# utils/yolo_export.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pymongo import ASCENDING

# Must match the app's BOX_CLASSES ordering EXACTLY
BOX_CLASSES = ["plate", "bike", "rider_head", "rider_hand", "traffic_signal"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_yolo_indexes(db, collection_name: str = "yolo_labels") -> None:
    """
    One YOLO label doc per *frame*.
    Uniqueness must include video_id + frame_file (frame_file is only unique within a video).
    """
    col = db[collection_name]
    col.create_index([("video_id", ASCENDING), ("frame_file", ASCENDING)], unique=True)
    col.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)])
    col.create_index([("video_id", ASCENDING), ("frame_status", ASCENDING)])
    col.create_index([("video_id", ASCENDING), ("has_boxes", ASCENDING)])
    col.create_index([("video_id", ASCENDING), ("has_violations", ASCENDING)])


def _class_id(label: str, class_list: List[str]) -> Optional[int]:
    label = (label or "").strip()
    try:
        return class_list.index(label)
    except ValueError:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _frame_files_for_clip(db, video_id: str, clip_id: str) -> List[str]:
    """
    Clip doc is the source of truth for frame_files ordering.
    """
    clip = db.clips.find_one({"video_id": video_id, "clip_id": clip_id}, {"_id": 0, "frame_files": 1, "frames": 1}) or {}
    frame_files = clip.get("frame_files") or []
    if frame_files:
        return frame_files

    # Fallback (should not happen if ingest is correct)
    n_frames = len(clip.get("frames", []) or [])
    return [f"frame_{i:08d}.jpg" for i in range(n_frames)]


def upsert_yolo_labels_for_clip(
    db,
    clip_rec: Dict[str, Any],
    *,
    class_list: List[str] = BOX_CLASSES,
    collection_name: str = "yolo_labels",
    include_discarded: bool = False,
) -> Dict[str, Any]:
    """
    Upsert YOLO labels ONLY for one clip record (fast).
    Expects clip_rec fields:
      - video_id, clip_id, frames (list[str])
      - frame_status_map (dict), frame_tags_map (dict)
      - boxes (list[dict] with frame,label,x,y,w,h)
    """
    ensure_yolo_indexes(db, collection_name=collection_name)
    out_col = db[collection_name]

    video_id = clip_rec["video_id"]
    clip_id = clip_rec["clip_id"]
    frame_files = clip_rec.get("frames") or []

    frame_status_map = clip_rec.get("frame_status_map") or {}
    frame_tags_map = clip_rec.get("frame_tags_map") or {}
    boxes = clip_rec.get("boxes") or []

    # Group boxes by frame
    boxes_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for b in boxes:
        fn = b.get("frame")
        if fn:
            boxes_by_frame.setdefault(fn, []).append(b)

    written = 0
    skipped = 0
    unknown_labels = 0

    for fi, frame_file in enumerate(frame_files):
        frame_status = frame_status_map.get(frame_file, "usable")
        if (frame_status != "usable") and (not include_discarded):
            skipped += 1
            # Optional: still write an empty doc or delete existing
            out_col.delete_one({"video_id": video_id, "frame_file": frame_file})
            continue

        frame_tags = frame_tags_map.get(frame_file, []) or []
        frame_boxes = boxes_by_frame.get(frame_file, []) or []

        yolo_lines: List[str] = []
        for b in frame_boxes:
            label = str(b.get("label", "")).strip()
            cid = _class_id(label, class_list)
            if cid is None:
                unknown_labels += 1
                continue

            x = float(b.get("x", 0.0))
            y = float(b.get("y", 0.0))
            w = float(b.get("w", 0.0))
            h = float(b.get("h", 0.0))

            # clamp
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            yolo_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        doc = {
            "video_id": video_id,
            "clip_id": clip_id,
            "frame_file": frame_file,
            "frame_index": fi,
            "frame_status": frame_status,
            "frame_tags": frame_tags,
            "yolo_lines": yolo_lines,
            "yolo_txt": "\n".join(yolo_lines),
            "updated_at": now_iso(),
        }

        out_col.update_one(
            {"video_id": video_id, "frame_file": frame_file},
            {"$set": doc},
            upsert=True,
        )
        written += 1

    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "frames_written": written,
        "frames_skipped_nonusable": skipped,
        "unknown_label_count": unknown_labels,
        "collection": collection_name,
    }



def export_yolo_to_mongo(
    db,
    video_id: str,
    *,
    class_list: List[str] = BOX_CLASSES,
    collection_name: str = "yolo_labels",
    include_nonusable: bool = False,
) -> Dict[str, Any]:
    """
    SLOW PATH: export YOLO docs for the entire video (all clips).
    Use this for batch jobs / rebuilds, NOT for every UI save.
    """
    ensure_yolo_indexes(db, collection_name=collection_name)
    out_col = db[collection_name]

    clips = list(db.clips.find({"video_id": video_id}, {"_id": 0}).sort("clip_id", 1))

    # annotation map
    ann_map: Dict[str, Dict[str, Any]] = {}
    for a in db.annotations.find({"video_id": video_id}, {"_id": 0}):
        ann_map[str(a.get("clip_id"))] = a

    total_frames_seen = 0
    frames_written = 0
    frames_skipped_nonusable = 0
    unknown_label_count = 0

    for clip in clips:
        clip_id = str(clip.get("clip_id"))
        ann = ann_map.get(clip_id, {}) or {}
        res = upsert_yolo_labels_for_clip(
            db,
            ann_rec=ann,
            class_list=class_list,
            collection_name=collection_name,
            include_nonusable=include_nonusable,
        )
        # res is per-clip; update totals roughly
        frame_files = clip.get("frame_files") or []
        if not frame_files:
            n_frames = len(clip.get("frames", []) or [])
            frame_files = [f"frame_{i:08d}.jpg" for i in range(n_frames)]
        total_frames_seen += len(frame_files)

        if res.get("ok"):
            frames_written += int(res.get("frames_written", 0))
            frames_skipped_nonusable += int(res.get("frames_skipped_nonusable", 0))
            unknown_label_count += int(res.get("unknown_label_count", 0))

    return {
        "video_id": video_id,
        "collection": collection_name,
        "class_list": class_list,
        "include_nonusable": include_nonusable,
        "total_frames_seen": total_frames_seen,
        "frames_written": frames_written,
        "frames_skipped_nonusable": frames_skipped_nonusable,
        "unknown_label_count": unknown_label_count,
    }
