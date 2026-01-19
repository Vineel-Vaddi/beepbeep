"""
Bulk YouTube -> Frames extractor (with goals + logs)

Features
1) Interactive menu:
   A. CSV of YouTube URLs
   B. Single YouTube URL
   C. Extract frames from already-downloaded video(s)
2) Goal selection: Fast labeling / Balanced / High detail
3) Automatically chooses EVERY_N_FRAMES based on FPS and goal
4) Per-video folders under downloads/ and frames_out/
5) Appends run logs to 1 CSV + 1 JSONL file

Install:
  pip install -U yt-dlp opencv-python

Run:
  python bulk_yt_to_frames.py
"""

import os
import re
import csv
import json
import cv2
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple

# -----------------------------
# Config
# -----------------------------
DOWNLOADS_ROOT = "downloads"
FRAMES_ROOT = "frames_out"
LOGS_DIR = "logs"
LOG_CSV = os.path.join(LOGS_DIR, "runs_log.csv")
LOG_JSONL = os.path.join(LOGS_DIR, "runs_log.jsonl")  # newline-delimited JSON

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}

# Goal -> target fps (frames saved per second)
GOAL_TARGET_FPS = {
    "fast": 2.0,      # fewer frames, faster labeling
    "balanced": 3.0,  # recommended starting point
    "detail": 4.0,    # more frames, more labeling
}

# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    s = s.strip().replace(" ", "_")
    return s[:120] if len(s) > 120 else s

def ensure_dirs():
    Path(DOWNLOADS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(FRAMES_ROOT).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

def now_ts() -> Tuple[str, str]:
    dt = datetime.now()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")

def pick_every_n_frames(fps: float, goal_key: str) -> int:
    """
    Decide EVERY_N_FRAMES based on measured FPS and selected goal.
    Rule: every_n = round(fps / target_fps). Minimum 1.
    """
    target = GOAL_TARGET_FPS[goal_key]
    every_n = int(round(float(fps) / float(target)))
    return max(1, every_n)

def get_video_meta(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": float(fps), "total_frames": int(total)}

def download_youtube(url: str, out_dir: str) -> str:
    """
    Download best mp4 using yt-dlp into out_dir.
    Returns downloaded file path.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_template = os.path.join(out_dir, "%(title)s_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url
    ]

    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    mp4_files = sorted(Path(out_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4_files:
        raise FileNotFoundError(f"Download finished but no .mp4 found in {out_dir}")
    return str(mp4_files[0])

def get_title_from_filename(video_path: str) -> str:
    # yt-dlp file template usually includes title_id; we can use stem as "title-ish"
    return Path(video_path).stem

def extract_frames(
    video_path: str,
    out_dir: str,
    every_n_frames: int,
    max_frames: Optional[int] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    resize_width: Optional[int] = 1280,
    jpg_quality: int = 95
) -> int:
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

    print(f"\nVideo: {video_path}")
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
    return saved

def read_urls_from_csv(csv_path: str) -> List[str]:
    """
    Reads a CSV and extracts URLs from first column by default.
    If there's a header, it still works.
    """
    urls = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = row[0].strip()
            if not cell:
                continue
            # Basic filter: keep likely YouTube links
            if "youtube.com" in cell or "youtu.be" in cell:
                urls.append(cell)
    # de-dup preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def init_log_files():
    ensure_dirs()
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "date", "time",
                "source_mode", "goal",
                "yt_url",
                "video_title",
                "downloaded_video_path",
                "frames_output_folder",
                "fps", "total_frames",
                "every_n_frames",
                "frames_extracted",
                "status",
                "error"
            ])
            w.writeheader()

def append_log(record: Dict):
    # CSV
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "time",
            "source_mode", "goal",
            "yt_url",
            "video_title",
            "downloaded_video_path",
            "frames_output_folder",
            "fps", "total_frames",
            "every_n_frames",
            "frames_extracted",
            "status",
            "error"
        ])
        w.writerow(record)

    # JSONL
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_video_folders(video_id_hint: str) -> Tuple[str, str]:
    """
    Create per-video folder names under downloads/ and frames_out/.
    video_id_hint can be URL id or filename stem.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = safe_name(f"{video_id_hint}_{stamp}")
    dl_dir = os.path.join(DOWNLOADS_ROOT, folder)
    fr_dir = os.path.join(FRAMES_ROOT, folder)
    Path(dl_dir).mkdir(parents=True, exist_ok=True)
    Path(fr_dir).mkdir(parents=True, exist_ok=True)
    return dl_dir, fr_dir

def extract_yt_id(url: str) -> str:
    # Best-effort YouTube id extraction
    m = re.search(r"v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    return safe_name(url)[:20]

def choose_goal() -> str:
    print("\nSelect Goal:")
    print("  1) Fast labeling (≈2 fps)")
    print("  2) Balanced (≈3 fps) [recommended]")
    print("  3) High detail (≈4 fps)")
    while True:
        c = input("Enter choice (1/2/3): ").strip()
        if c == "1":
            return "fast"
        if c == "2":
            return "balanced"
        if c == "3":
            return "detail"
        print("Invalid choice. Try again.")

def choose_mode() -> str:
    print("\nChoose input mode:")
    print("  1) Choose CSV file (list of YouTube URLs)")
    print("  2) Provide single YouTube URL")
    print("  3) Convert to frames from already downloaded video (file or folder)")
    while True:
        c = input("Enter choice (1/2/3): ").strip()
        if c in {"1", "2", "3"}:
            return c
        print("Invalid choice. Try again.")

def list_videos_in_path(path_str: str) -> List[str]:
    p = Path(path_str)
    if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
        return [str(p)]
    if p.is_dir():
        vids = []
        for ext in SUPPORTED_VIDEO_EXTS:
            vids.extend([str(x) for x in p.rglob(f"*{ext}")])
        vids.sort()
        return vids
    return []

# -----------------------------
# Main pipelines
# -----------------------------
def process_youtube_urls(urls: List[str], goal_key: str, source_mode: str):
    for idx, url in enumerate(urls, start=1):
        date_s, time_s = now_ts()

        yt_id = extract_yt_id(url)
        dl_dir, fr_dir = make_video_folders(yt_id)

        record = {
            "date": date_s,
            "time": time_s,
            "source_mode": source_mode,
            "goal": goal_key,
            "yt_url": url,
            "video_title": "",
            "downloaded_video_path": "",
            "frames_output_folder": fr_dir,
            "fps": "",
            "total_frames": "",
            "every_n_frames": "",
            "frames_extracted": "",
            "status": "STARTED",
            "error": ""
        }

        print(f"\n\n==============================")
        print(f"[{idx}/{len(urls)}] Processing URL: {url}")
        print(f"Downloads folder: {dl_dir}")
        print(f"Frames folder   : {fr_dir}")
        print(f"==============================")

        try:
            video_path = download_youtube(url, out_dir=dl_dir)
            title = get_title_from_filename(video_path)

            meta = get_video_meta(video_path)
            fps = meta["fps"]
            total_frames = meta["total_frames"]

            every_n = pick_every_n_frames(fps, goal_key)

            frames_saved = extract_frames(
                video_path=video_path,
                out_dir=fr_dir,
                every_n_frames=every_n,
                max_frames=None,
                start_sec=0,
                end_sec=None,
                resize_width=1280
            )

            record.update({
                "video_title": title,
                "downloaded_video_path": video_path,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "every_n_frames": every_n,
                "frames_extracted": frames_saved,
                "status": "SUCCESS",
                "error": ""
            })

        except Exception as e:
            record.update({
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Error for URL {url}: {e}")

        append_log(record)

def process_local_videos(video_paths: List[str], goal_key: str, source_mode: str):
    for idx, video_path in enumerate(video_paths, start=1):
        date_s, time_s = now_ts()

        title = get_title_from_filename(video_path)
        hint = safe_name(title)

        # Keep local files where they are; still create per-run output folders
        _, fr_dir = make_video_folders(hint)

        record = {
            "date": date_s,
            "time": time_s,
            "source_mode": source_mode,
            "goal": goal_key,
            "yt_url": "",  # not applicable
            "video_title": title,
            "downloaded_video_path": video_path,
            "frames_output_folder": fr_dir,
            "fps": "",
            "total_frames": "",
            "every_n_frames": "",
            "frames_extracted": "",
            "status": "STARTED",
            "error": ""
        }

        print(f"\n\n==============================")
        print(f"[{idx}/{len(video_paths)}] Processing local video: {video_path}")
        print(f"Frames folder: {fr_dir}")
        print(f"==============================")

        try:
            meta = get_video_meta(video_path)
            fps = meta["fps"]
            total_frames = meta["total_frames"]

            every_n = pick_every_n_frames(fps, goal_key)

            frames_saved = extract_frames(
                video_path=video_path,
                out_dir=fr_dir,
                every_n_frames=every_n,
                max_frames=None,
                start_sec=0,
                end_sec=None,
                resize_width=1280
            )

            record.update({
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "every_n_frames": every_n,
                "frames_extracted": frames_saved,
                "status": "SUCCESS",
                "error": ""
            })

        except Exception as e:
            record.update({
                "status": "FAILED",
                "error": str(e)
            })
            print(f"❌ Error for local video {video_path}: {e}")

        append_log(record)

def main():
    init_log_files()

    mode = choose_mode()
    goal_key = choose_goal()

    if mode == "1":
        csv_path = input("\nEnter CSV file path (with YouTube URLs in first column): ").strip().strip('"')
        if not os.path.exists(csv_path):
            print("❌ CSV file not found.")
            return
        urls = read_urls_from_csv(csv_path)
        if not urls:
            print("❌ No YouTube URLs found in the CSV.")
            return
        print(f"✅ Found {len(urls)} URL(s) in CSV.")
        process_youtube_urls(urls, goal_key, source_mode="CSV")

    elif mode == "2":
        url = input("\nPaste YouTube URL: ").strip()
        if not url:
            print("❌ Empty URL.")
            return
        process_youtube_urls([url], goal_key, source_mode="SINGLE_URL")

    elif mode == "3":
        path_str = input("\nEnter local video file path OR folder path containing videos: ").strip().strip('"')
        videos = list_videos_in_path(path_str)
        if not videos:
            print("❌ No video files found at the given path.")
            return
        print(f"✅ Found {len(videos)} local video(s).")
        process_local_videos(videos, goal_key, source_mode="LOCAL_VIDEO")

    print("\n✅ All done.")
    print(f"Logs saved to:\n  - {LOG_CSV}\n  - {LOG_JSONL}")

if __name__ == "__main__":
    main()
