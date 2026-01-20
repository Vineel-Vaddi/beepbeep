# yolo_export.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

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


def export_yolo_to_mongo(
    db,
    video_id: str,
    *,
    class_list: List[str] = BOX_CLASSES,
    collection_name: str = "yolo_labels",
    include_nonusable: bool = False,
) -> Dict[str, Any]:
    """
    Reads:
      - db.clips        : clip_id, frame_files ordering
      - db.annotations  : frame_status_map, frame_tags_map, boxes (normalized x,y,w,h)

    Writes:
      - db[collection_name] : one doc per frame with YOLO lines (+ helpful flags)

    IMPORTANT (matches your new approach):
      - Tags are FRAME-level: frame_tags_map[frame_file] is used per frame
      - Boxes are FRAME-level: each box has "frame": frame_file
      - Frames can be triaged: frame_status_map[frame_file] in {untriaged, usable, discarded, unclear}

    include_nonusable:
      - False => export only usable frames
      - True  => export all frames, but keep frame_status so you can filter later
    """
    ensure_yolo_indexes(db, collection_name=collection_name)
    out_col = db[collection_name]

    # Load all clips for this video
    clips = list(db.clips.find({"video_id": video_id}, {"_id": 0}).sort("clip_id", 1))

    # Load all annotations for this video into map: clip_id -> annotation doc
    ann_map: Dict[str, Dict[str, Any]] = {}
    for a in db.annotations.find({"video_id": video_id}, {"_id": 0}):
        ann_map[str(a.get("clip_id"))] = a

    total_frames_seen = 0
    frames_written = 0
    frames_skipped_nonusable = 0
    unknown_label_count = 0
    missing_frame_files_fallback = 0

    for clip in clips:
        clip_id = str(clip.get("clip_id"))
        frame_files = clip.get("frame_files") or []

        if not frame_files:
            # fallback (should not happen if ingest is correct)
            n_frames = len(clip.get("frames", []) or [])
            frame_files = [f"frame_{i:08d}.jpg" for i in range(n_frames)]
            missing_frame_files_fallback += 1

        ann = ann_map.get(clip_id, {}) or {}
        frame_status_map = ann.get("frame_status_map") or {}
        frame_tags_map = ann.get("frame_tags_map") or {}
        boxes = ann.get("boxes") or []

        # Group boxes by frame filename
        boxes_by_frame: Dict[str, List[Dict[str, Any]]] = {}
        for b in boxes:
            fn = b.get("frame")
            if not fn:
                continue
            boxes_by_frame.setdefault(fn, []).append(b)

        for fi, frame_file in enumerate(frame_files):
            total_frames_seen += 1

            frame_status = (frame_status_map.get(frame_file) or "untriaged").strip().lower()

            # Export only usable frames by default
            if (frame_status != "usable") and (not include_nonusable):
                frames_skipped_nonusable += 1
                continue

            frame_tags = frame_tags_map.get(frame_file, []) or []
            frame_boxes = boxes_by_frame.get(frame_file, []) or []

            yolo_lines: List[str] = []
            for b in frame_boxes:
                cid = _class_id(str(b.get("label", "")).strip(), class_list)
                if cid is None:
                    unknown_label_count += 1
                    continue

                # already normalized YOLO center coords in app
                x = _clamp01(b.get("x", 0.0))
                y = _clamp01(b.get("y", 0.0))
                w = _clamp01(b.get("w", 0.0))
                h = _clamp01(b.get("h", 0.0))

                # Optional: ignore degenerate boxes
                if w <= 0.0 or h <= 0.0:
                    continue

                yolo_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            has_boxes = len(yolo_lines) > 0
            has_violations = len(frame_tags) > 0

            doc = {
                "video_id": video_id,
                "clip_id": clip_id,
                "frame_file": frame_file,
                "frame_index": int(fi),

                # triage + tags (frame-level)
                "frame_status": frame_status,  # usable / discarded / unclear / untriaged
                "frame_tags": frame_tags,

                # YOLO output
                "yolo_lines": yolo_lines,
                "yolo_txt": "\n".join(yolo_lines),

                # convenience flags for filtering/training
                "has_boxes": has_boxes,
                "has_violations": has_violations,

                "updated_at": now_iso(),
            }

            # IMPORTANT: uniqueness must include video_id
            out_col.update_one(
                {"video_id": video_id, "frame_file": frame_file},
                {"$set": doc},
                upsert=True,
            )
            frames_written += 1

    return {
        "video_id": video_id,
        "collection": collection_name,
        "class_list": class_list,
        "include_nonusable": include_nonusable,
        "total_frames_seen": total_frames_seen,
        "frames_written": frames_written,
        "frames_skipped_nonusable": frames_skipped_nonusable,
        "unknown_label_count": unknown_label_count,
        "missing_frame_files_fallback": missing_frame_files_fallback,
    }
