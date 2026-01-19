import os
import re
import cv2
import subprocess
from pathlib import Path

def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    s = s.strip().replace(" ", "_")
    return s[:120] if len(s) > 120 else s

def download_youtube(url: str, out_dir: str = "downloads") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_template = os.path.join(out_dir, "%(title)s_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    mp4_files = sorted(Path(out_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4_files:
        raise FileNotFoundError("Download finished but no .mp4 found in downloads/")

    return str(mp4_files[0])

def extract_frames(
    video_path: str,
    out_dir: str = "frames",
    every_n_frames: int = 10,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    resize_width: int | None = 1280,
    jpg_quality: int = 95
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else total - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_idx = start_frame

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total}")
    print(f"Extracting frames {start_frame} to {end_frame}, saving every {every_n_frames} frames...")

    while True:
        if end_sec is not None and frame_idx > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % every_n_frames == 0:
            if resize_width is not None and resize_width > 0:
                h, w = frame.shape[:2]
                new_w = int(resize_width)
                new_h = int(h * (new_w / w))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:08d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
            saved += 1

            if saved % 50 == 0:
                print(f"Saved {saved} frames... (latest: {out_path})")

            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"✅ Done. Saved {saved} frames to: {out_dir}")

if __name__ == "__main__":
    # ==== EDIT THESE ====
    YT_URL = "https://www.youtube.com/watch?v=4eKGek1y2zM"
    EVERY_N_FRAMES = 10        # 10 => ~3 fps for 30fps video
    START_SEC = 0
    END_SEC = None            # e.g., 120 for first 2 minutes
    MAX_FRAMES = None         # e.g., 1500 to limit
    RESIZE_WIDTH = 1280       # None to keep original
    # ====================

    video_file = download_youtube(YT_URL, out_dir="downloads")

    video_stem = safe_name(Path(video_file).stem)
    out_frames = os.path.join("frames_out", video_stem)

    extract_frames(
        video_path=video_file,
        out_dir=out_frames,
        every_n_frames=EVERY_N_FRAMES,
        max_frames=MAX_FRAMES,
        start_sec=START_SEC,
        end_sec=END_SEC,
        resize_width=RESIZE_WIDTH
    )
