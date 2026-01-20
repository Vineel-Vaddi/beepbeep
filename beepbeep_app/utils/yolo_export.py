from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional

from pymongo import ASCENDING


# Must match the app's BOX_CLASSES ordering
BOX_CLASSES = ["plate", "bike", "rider_head", "rider_hand", "traffic_signal"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_yolo_indexes(db, collection_name: str = "yolo_labels") -> None:
    """
    One YOLO label doc per frame.
    """
    col = db[collection_name]
    col.create_index([("video_id", ASCENDING), ("frame_file", ASCENDING)], unique=True)
    col.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)])
    col.create_index([("video_id", ASCENDING), ("frame_status", ASCENDING)])


def _class_id(label: str, class_list: List[str]) -> Optional[int]:
    try:
        return class_list.index(label)
    except ValueError:
        return None


def export_yolo_to_mongo(
    db,
    video_id: str,
    *,
    class_list: List[str] = BOX_CLASSES,
    collection_name: str = "yolo_labels",
    include_discarded: bool = False,
) -> Dict[str, Any]:
    """
    Reads:
      - db.clips (for frame_files ordering)
      - db.annotations (for frame_status_map, frame_tags_map, boxes)

    Writes:
      - db[collection_name] one doc per frame with YOLO lines

    Returns summary counts.
    """
    ensure_yolo_indexes(db, collection_name=collection_name)
    out_col = db[collection_name]

    clips = list(db.clips.find({"video_id": video_id}, {"_id": 0}).sort("clip_id", 1))

    # Build annotation map clip_id -> annotation doc
    ann_map: Dict[str, Dict[str, Any]] = {}
    for a in db.annotations.find({"video_id": video_id}, {"_id": 0}):
        ann_map[a["clip_id"]] = a

    total_frames = 0
    written = 0
    skipped = 0
    unknown_labels = 0

    for clip in clips:
        clip_id = clip["clip_id"]
        frame_files = clip.get("frame_files") or []
        if not frame_files:
            # fallback if not present
            n_frames = len(clip.get("frames", []) or [])
            frame_files = [f"frame_{i:06d}.jpg" for i in range(n_frames)]

        ann = ann_map.get(clip_id, {})
        frame_status_map = ann.get("frame_status_map") or {}
        frame_tags_map = ann.get("frame_tags_map") or {}
        boxes = ann.get("boxes") or []

        # Group boxes by frame
        boxes_by_frame: Dict[str, List[Dict[str, Any]]] = {}
        for b in boxes:
            fn = b.get("frame")
            if not fn:
                continue
            boxes_by_frame.setdefault(fn, []).append(b)

        for fi, frame_file in enumerate(frame_files):
            total_frames += 1

            frame_status = frame_status_map.get(frame_file, "usable")
            if (frame_status != "usable") and (not include_discarded):
                skipped += 1
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

                # already normalized center coords
                x = float(b.get("x", 0.0))
                y = float(b.get("y", 0.0))
                w = float(b.get("w", 0.0))
                h = float(b.get("h", 0.0))

                # clamp (safety)
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
        "total_frames_seen": total_frames,
        "frames_written": written,
        "frames_skipped_nonusable": skipped,
        "unknown_label_count": unknown_labels,
        "collection": collection_name,
        "class_list": class_list,
    }
