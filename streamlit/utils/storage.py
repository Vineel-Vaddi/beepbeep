import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


def ensure_video_annotation_dir(video_folder: str) -> str:
    base = "annotations"
    ann_dir = os.path.join(base, video_folder)
    os.makedirs(ann_dir, exist_ok=True)
    return ann_dir


def _jsonl_path(ann_dir: str) -> str:
    return os.path.join(ann_dir, "annotations.jsonl")


def _state_path(ann_dir: str) -> str:
    return os.path.join(ann_dir, "state.json")


def append_annotation_record(ann_dir: str, record: Dict[str, Any]) -> None:
    """
    Safe append-only JSONL.
    Crash-safe: each line is an independent JSON object.
    """
    os.makedirs(ann_dir, exist_ok=True)
    p = _jsonl_path(ann_dir)
    line = json.dumps(record, ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_latest_annotations_map(ann_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Read JSONL and keep the latest record per clip_id.
    If file is large, this is still okay Phase-1; later can index.
    """
    p = _jsonl_path(ann_dir)
    latest: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(p):
        return latest

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("clip_id")
                if cid is None:
                    continue
                latest[cid] = rec
            except Exception:
                # ignore corrupted line (rare). append-only reduces risk.
                continue
    return latest


def load_state(ann_dir: str) -> Dict[str, Any]:
    p = _state_path(ann_dir)
    if not os.path.isfile(p):
        return {"cursor_clip_idx": 0}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"cursor_clip_idx": 0}


def save_state(ann_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(ann_dir, exist_ok=True)
    p = _state_path(ann_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, p)


def compute_progress_counts(clips: List[Dict[str, Any]], ann_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    done = skipped = unclear = unlabeled = 0
    for c in clips:
        cid = c["clip_id"]
        rec = ann_map.get(cid)
        status = rec.get("review_status", "unlabeled") if rec else "unlabeled"
        if status == "done":
            done += 1
        elif status == "skipped":
            skipped += 1
        elif status == "unclear":
            unclear += 1
        else:
            unlabeled += 1
    return {"done": done, "skipped": skipped, "unclear": unclear, "unlabeled": unlabeled}


def get_last_labeled_clip_idx(clips: List[Dict[str, Any]], ann_map: Dict[str, Dict[str, Any]]) -> Optional[int]:
    last = None
    for i, c in enumerate(clips):
        cid = c["clip_id"]
        rec = ann_map.get(cid)
        status = rec.get("review_status", "unlabeled") if rec else "unlabeled"
        if status != "unlabeled":
            last = i
    return last
