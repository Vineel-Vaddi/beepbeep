# Traffic Violation Labeler — Phase-1 (Streamlit)

## What it does
- Loads frames from: `frames_out/<video_folder>/frame_*.jpg`
- Near-duplicate filtering via dHash (keeps only sufficiently different frames)
- Groups frames into clips (3–6 frames, default 4)
- Clip-level violation tags (multi-label)
- Per-frame bounding boxes (stored normalized YOLO-style center x,y,w,h)
- Autosave on every action + resume cursor
- Append-only JSONL per video (crash-safe)

## Install
```bash
pip install -r requirements.txt
