import os
import cv2
from pathlib import Path
from typing import List


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: int = 2
) -> List[str]:
    """
    Extract frames from a video at a fixed interval.

    Args:
        video_path: Path to input video
        output_dir: Folder where extracted frames will be saved
        interval_seconds: Extract one frame every N seconds

    Returns:
        List of saved frame file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback if FPS is unreadable

    frame_interval = int(fps * interval_seconds)
    if frame_interval <= 0:
        frame_interval = 1

    saved_frames = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = output_path / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_frames.append(str(frame_filename))
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_frames