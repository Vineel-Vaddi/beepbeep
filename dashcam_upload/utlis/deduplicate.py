from pathlib import Path
from typing import List, Tuple
from PIL import Image
import imagehash


def remove_duplicate_frames(
    frame_paths: List[str],
    hash_size: int = 8,
    duplicate_threshold: int = 5
) -> Tuple[List[str], List[str]]:
    """
    Remove near-duplicate frames using perceptual hashing.

    Args:
        frame_paths: List of extracted frame file paths
        hash_size: Size used for perceptual hash
        duplicate_threshold: Max hash difference to consider duplicate

    Returns:
        (unique_frames, removed_frames)
    """
    unique_frames = []
    removed_frames = []
    seen_hashes = []

    for frame_path in frame_paths:
        try:
            img = Image.open(frame_path)
            current_hash = imagehash.phash(img, hash_size=hash_size)

            is_duplicate = False
            for prev_hash in seen_hashes:
                if abs(current_hash - prev_hash) <= duplicate_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                removed_frames.append(frame_path)
                Path(frame_path).unlink(missing_ok=True)
            else:
                unique_frames.append(frame_path)
                seen_hashes.append(current_hash)

        except Exception:
            removed_frames.append(frame_path)
            Path(frame_path).unlink(missing_ok=True)

    return unique_frames, removed_frames