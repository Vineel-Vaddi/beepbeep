import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image

import streamlit as st


def list_video_folders(dataset_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(dataset_root)):
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p):
            out.append(name)
    return out


def _sorted_frame_files(frames_dir: str) -> List[str]:
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Difference hash (dHash). Returns integer bit-hash.
    """
    img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    # compare adjacent pixels row-wise
    diff = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            diff.append(left > right)
    # convert bits to int
    v = 0
    for bit in diff:
        v = (v << 1) | int(bit)
    return v


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_or_load_index(frames_dir: str, ann_dir: str, dhash_threshold: int = 6) -> Dict[str, Any]:
    """
    Build (or load) index.json containing near-duplicate filtered frames list.
    Keeps frame if hamming(dhash, last_kept_hash) >= threshold.
    """
    os.makedirs(ann_dir, exist_ok=True)
    index_path = os.path.join(ann_dir, "index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    files = _sorted_frame_files(frames_dir)
    kept = []
    dropped = []
    mapping = {}  # kept -> original (same here, but reserved for future trace)

    last_hash = None
    for f in files:
        p = os.path.join(frames_dir, f)
        try:
            img = Image.open(p)
            hv = dhash(img)
        except Exception:
            dropped.append(f)
            continue

        if last_hash is None:
            kept.append(f)
            mapping[f] = f
            last_hash = hv
        else:
            if hamming(last_hash, hv) >= dhash_threshold:
                kept.append(f)
                mapping[f] = f
                last_hash = hv
            else:
                dropped.append(f)

    index_data = {
        "frames_dir": frames_dir,
        "kept_frames": kept,
        "dropped_frames": dropped,
        "mapping": mapping,
        "dhash_threshold": dhash_threshold,
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    return index_data


def build_clips_from_index(index_data: Dict[str, Any], clip_len: int = 4) -> List[Dict[str, Any]]:
    """
    Group kept frames into clips of length 3-6 (default 4).
    Non-overlapping grouping for speed.
    """
    frames = index_data["kept_frames"]
    clip_len = max(3, min(6, int(clip_len)))

    clips = []
    clip_idx = 0
    for i in range(0, len(frames), clip_len):
        group = frames[i : i + clip_len]
        if not group:
            continue
        clips.append(
            {
                "clip_id": f"{clip_idx:06d}",
                "frames": group,
            }
        )
        clip_idx += 1
    return clips


def get_clip_frames(clips: List[Dict[str, Any]], clip_idx: int) -> List[str]:
    return clips[clip_idx]["frames"]


@st.cache_data(show_spinner=False)
def load_image_cached(path: str, max_w: int = 1100) -> Image.Image:
    """
    Cached image load + optional downscale for speed.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
    return img
