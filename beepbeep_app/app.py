import json
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import base64
import io
import plotly.graph_objects as go
from PIL import Image



from utils.mongo_backend import (
    get_db,
    ensure_indexes,
    list_videos,
    get_video_stats,
    load_state,
    save_state,
    load_clips,
    load_annotations_map,
    upsert_annotation,
    compute_progress_counts,
    get_last_labeled_clip_idx,
    load_frame_image_cached,
)

# -----------------------------
# Fixed label sets (Phase-1)
# -----------------------------
VIOLATION_TAGS = ["no_helmet", "triple_riding", "signal_jump", "phone_driving"]
BOX_CLASSES = ["plate", "bike", "rider_head", "rider_hand", "traffic_signal"]

# Color per box class (RGBA with transparency)
BOX_CLASS_STYLE = {
    "plate": "rgba(255, 215, 0, 0.25)",          # gold
    "bike": "rgba(0, 176, 246, 0.25)",           # blue
    "rider_head": "rgba(0, 200, 83, 0.25)",      # green
    "rider_hand": "rgba(255, 87, 34, 0.25)",     # orange
    "traffic_signal": "rgba(156, 39, 176, 0.25)" # purple
}


DEFAULT_ANNOTATOR = "user1"
FILTER_MODES = ["All", "Needs Triage", "Needs Tags", "Needs Boxes", "Complete", "Has Violations"]



def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_session():
    # Always create all keys (callbacks may fire early on rerun)
    defaults = {
        "page": "selector",                 # selector | labeling
        "video_folder": None,
        "annotator": DEFAULT_ANNOTATOR,

        "clips": None,
        "ann_map": {},

        "clip_idx": 0,

        "filter_mode": "All",
        "jump_to": 0,

        # Selected working frame inside the clip
        "frame_idx": 0,

        # Current frame boxes (only boxes for selected frame)
        "canvas_boxes": [],
        "last_canvas_hash": "",

        # Frame-level maps for current clip (loaded from Mongo)
        "frame_status_map": {},   # {frame_file: usable|discarded|unclear|done}
        "frame_tags_map": {},     # {frame_file: [tags...]}

        # Plotly persistence
        "plotly_fig_state": {},

        # UX helpers
        "active_box_class": BOX_CLASSES[0],

        "keyboard_last": "",

    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v




def load_project(video_folder: str):
    db = get_db()
    ensure_indexes(db)

    clips = load_clips(db, video_folder)  # expects clip docs with frames + frame_files
    ann_map = load_annotations_map(db, video_folder)
    state = load_state(db, video_folder)

    clip_idx = state.get("cursor_clip_idx", 0)
    clip_idx = max(0, min(clip_idx, max(0, len(clips) - 1)))

    st.session_state.video_folder = video_folder
    st.session_state.clips = clips
    st.session_state.ann_map = ann_map
    st.session_state.clip_idx = clip_idx
    st.session_state.filter_mode = state.get("filter_mode", "All")

    load_clip_into_ui()


def load_clip_into_ui():
    """Load frame-level triage/tags and current frame boxes for selected frame."""
    video_folder = st.session_state.video_folder
    if not video_folder:
        return
    clips = st.session_state.clips
    if not clips:
        return

    clip_idx = max(0, min(st.session_state.clip_idx, len(clips) - 1))
    clip = clips[clip_idx]
    clip_id = clip["clip_id"]

    rec = st.session_state.ann_map.get(clip_id, {}) or {}

    frame_files = clip.get("frame_files") or clip.get("frames_files") or []
    if not frame_files:
        frame_files = [f"frame_{i:06d}.jpg" for i in range(len(clip.get("frames", []) or []))]

    # IMPORTANT: copy so we don't mutate ann_map objects accidentally
    frame_status_map = dict(rec.get("frame_status_map") or {})
    frame_tags_map = dict(rec.get("frame_tags_map") or {})

    # Ensure every frame has keys
    for f in frame_files:
        frame_status_map.setdefault(f, "untriaged")  # default: untriaged
        frame_tags_map.setdefault(f, [])             # default: no tags

    st.session_state.frame_status_map = frame_status_map
    st.session_state.frame_tags_map = frame_tags_map

    # Pick default frame_idx (triage-first)
    prev_idx = int(st.session_state.get("frame_idx", 0))
    if 0 <= prev_idx < len(frame_files):
        st.session_state.frame_idx = prev_idx
    else:
        untriaged_idxs = [i for i, f in enumerate(frame_files) if frame_status_map.get(f) == "untriaged"]
        usable_idxs = [i for i, f in enumerate(frame_files) if frame_status_map.get(f) == "usable"]
        st.session_state.frame_idx = untriaged_idxs[0] if untriaged_idxs else (usable_idxs[0] if usable_idxs else 0)

    refresh_boxes_for_selected_frame()
    autosave_state_only()



def refresh_boxes_for_selected_frame():
    """Load boxes for current clip & selected frame into session_state.canvas_boxes."""
    clips = st.session_state.clips
    if not clips:
        return

    clip = clips[st.session_state.clip_idx]
    clip_id = clip["clip_id"]

    frame_files = clip.get("frame_files") or clip.get("frames_files") or []
    if not frame_files:
        frame_files = [f"frame_{i:06d}.jpg" for i in range(len(clip.get("frames", []) or []))]

    if not frame_files:
        st.session_state.canvas_boxes = []
        st.session_state.last_canvas_hash = ""
        return

    i = int(st.session_state.get("frame_idx", 0))
    i = max(0, min(i, len(frame_files) - 1))
    frame_name = frame_files[i]

    rec = st.session_state.ann_map.get(clip_id, {}) or {}
    boxes = rec.get("boxes", []) or []

    boxes_f = [b for b in boxes if b.get("frame") == frame_name]

    st.session_state.canvas_boxes = boxes_f
    st.session_state.last_canvas_hash = json.dumps(boxes_f, sort_keys=True)


def current_clip() -> Dict[str, Any]:
    return st.session_state.clips[st.session_state.clip_idx]


def clip_record_from_ui() -> Dict[str, Any]:
    video_id = st.session_state.video_folder
    clip = current_clip()
    clip_id = clip["clip_id"]

    # Frame filenames used as stable keys everywhere
    frame_files = clip.get("frame_files") or clip.get("frames_files") or []
    if not frame_files:
        frame_files = [f"frame_{i:06d}.jpg" for i in range(len(clip.get("frames", [])) or [])]

    # Current selected frame in UI
    i = int(st.session_state.get("frame_idx", 0))
    i = max(0, min(i, max(0, len(frame_files) - 1)))
    frame_name = frame_files[i] if frame_files else ""

    existing = st.session_state.ann_map.get(clip_id, {})
    existing_boxes = existing.get("boxes", []) or []

    # Replace boxes only for selected frame; keep boxes for other frames
    other_boxes = [b for b in existing_boxes if b.get("frame") != frame_name]
    ui_boxes = st.session_state.canvas_boxes or []
    merged_boxes = other_boxes + ui_boxes

    # Frame-level maps (must be stored for YOLO export)
    frame_status_map = dict(st.session_state.get("frame_status_map") or {})
    frame_tags_map = dict(st.session_state.get("frame_tags_map") or {})

    # Ensure keys exist for every frame (sanity)
    # Recommended default: "untriaged" (NOT "usable")
    for f in frame_files:
        frame_status_map.setdefault(f, "untriaged")
        frame_tags_map.setdefault(f, [])

    # --- Derived helpers for filtering/progress ---
    usable_frames = [f for f in frame_files if frame_status_map.get(f) == "usable"]

    # union tags across usable frames (optional clip summary)
    all_tags = []
    for f in usable_frames:
        all_tags.extend(frame_tags_map.get(f, []) or [])
    clip_tags_union = sorted(set(all_tags))

    # boxes by frame for "needs boxes" logic
    boxes_by_frame = {}
    for b in merged_boxes:
        fn = b.get("frame")
        if fn:
            boxes_by_frame.setdefault(fn, []).append(b)

    def frame_needs_boxes(f: str) -> bool:
        # policy: if a frame has tags, it *may* need boxes.
        # If you want "only some tags require localization", add gating here.
        has_tags = len(frame_tags_map.get(f, []) or []) > 0
        has_boxes = len(boxes_by_frame.get(f, []) or []) > 0
        return has_tags and (not has_boxes)

    needs_triage = any(frame_status_map.get(f) == "untriaged" for f in frame_files)
    needs_tags = any((frame_status_map.get(f) == "usable") and (len(frame_tags_map.get(f, []) or []) == 0) for f in frame_files)
    needs_boxes = any((frame_status_map.get(f) == "usable") and frame_needs_boxes(f) for f in frame_files)

    # Derived review_status (used by compute_progress_counts)
    # Keep your labels: done/skipped/unclear/unlabeled
    if needs_triage or needs_tags or needs_boxes:
        review_status = "unlabeled"
    else:
        # If no usable frames, treat as skipped (everything discarded/unclear)
        if len(usable_frames) == 0:
            review_status = "skipped"
        else:
            review_status = "done"

    rec = {
        "video_id": video_id,
        "clip_id": clip_id,

        # keep frames list for reference/debug/export
        "frames": frame_files,

        # NEW (required for your Stage-A workflow + YOLO export)
        "frame_status_map": frame_status_map,
        "frame_tags_map": frame_tags_map,

        # boxes are stored per frame (each box includes "frame": "<frame_file>")
        "boxes": merged_boxes,

        # Optional clip summary (useful for quick filtering/search)
        "tags": clip_tags_union,
        "review_status": review_status,

        "annotator": st.session_state.get("annotator", DEFAULT_ANNOTATOR),
        "updated_at": now_iso(),

        # helpful cursor info
        "selected_frame_idx": i,
        "selected_frame": frame_name,
    }
    return rec



def autosave_annotation():
    """Upsert record to Mongo and update ann_map (latest wins)."""
    if not st.session_state.video_folder:
        return

    db = get_db()
    rec = clip_record_from_ui()

    upsert_annotation(db, rec)
    st.session_state.ann_map[rec["clip_id"]] = rec

    autosave_state_only()


def autosave_state_only():
    if not st.session_state.video_folder:
        return
    db = get_db()

    counts = compute_progress_counts(st.session_state.clips, st.session_state.ann_map)
    state = {
        "video_id": st.session_state.video_folder,
        "cursor_clip_idx": int(st.session_state.clip_idx),
        "filter_mode": st.session_state.filter_mode,
        "updated_at": now_iso(),
        "counts": counts,
    }
    save_state(db, st.session_state.video_folder, state)


def set_status(status: str):
    st.session_state.review_status = status
    autosave_annotation()


def clear_tags_toggleboxes():
    # keep sidebar checkboxes in sync when clearing via button
    for t in VIOLATION_TAGS:
        if f"tag_{t}" in st.session_state:
            st.session_state[f"tag_{t}"] = False


def clear_tags():
    st.session_state.tags = []
    clear_tags_toggleboxes()
    autosave_annotation()


def toggle_tag(tag: str):
    tags = set(st.session_state.tags)
    if tag in tags:
        tags.remove(tag)
    else:
        tags.add(tag)
    st.session_state.tags = sorted(list(tags))
    autosave_annotation()


def clear_boxes_current_keyframe():
    st.session_state.canvas_boxes = []
    autosave_annotation()


def copy_boxes_from_previous_clip():
    clips = st.session_state.clips
    idx = st.session_state.clip_idx
    if idx <= 0:
        return

    prev_clip = clips[idx - 1]
    prev_id = prev_clip["clip_id"]
    prev_rec = st.session_state.ann_map.get(prev_id)
    if not prev_rec:
        return

    cur = current_clip()
    cur_frame_files = cur.get("frame_files") or []
    if not cur_frame_files:
        cur_frame_files = [f"frame_{i:06d}.jpg" for i in range(len(cur.get("frames", [])))]

    i = max(0, min(st.session_state.frame_idx, len(cur_frame_files) - 1))
    frame_name = cur_frame_files[i]

    # Source frame from previous record (selected_frame if available)
    src_frame = prev_rec.get("selected_frame")
    prev_boxes = prev_rec.get("boxes", [])

    if src_frame:
        src = [b for b in prev_boxes if b.get("frame") == src_frame]
    else:
        src = prev_boxes[:]  # fallback

    copied = []
    for b in src:
        copied.append(
            {
                "frame": frame_name,
                "label": b.get("label", BOX_CLASSES[0]),
                "x": float(b.get("x", 0.5)),
                "y": float(b.get("y", 0.5)),
                "w": float(b.get("w", 0.1)),
                "h": float(b.get("h", 0.1)),
            }
        )

    st.session_state.canvas_boxes = copied
    autosave_annotation()



def propagate_boxes_to_all_frames_in_clip():
    """Optional helper: duplicate current keyframe boxes to all frames in clip (by filename keys)."""
    cur = current_clip()
    frame_files = cur.get("frame_files") or []
    if not frame_files:
        return

    i = max(0, min(st.session_state.frame_idx, len(frame_files) - 1))
    frame_name = frame_files[i]

    ui_boxes = st.session_state.canvas_boxes or []

    all_boxes = []
    for f in frame_files:
        for b in ui_boxes:
            all_boxes.append(
                {
                    "frame": f,
                    "label": b["label"],
                    "x": b["x"],
                    "y": b["y"],
                    "w": b["w"],
                    "h": b["h"],
                }
            )

    clip_id = cur["clip_id"]
    rec = st.session_state.ann_map.get(clip_id, {})
    rec_boxes_existing = rec.get("boxes", [])
    rec_boxes_nonclip = [b for b in rec_boxes_existing if b.get("frame") not in frame_files]
    st.session_state.ann_map[clip_id] = {**clip_record_from_ui(), "boxes": rec_boxes_nonclip + all_boxes, "selected_frame": frame_name}
    autosave_annotation()


def move_next_filtered(delta: int):
    """Move to next/prev clip respecting filter mode."""
    mode = st.session_state.filter_mode
    clips = st.session_state.clips
    n = len(clips)

    def ok(i: int) -> bool:
        clip_id = clips[i]["clip_id"]
        rec = st.session_state.ann_map.get(clip_id, {})

        frame_status_map = rec.get("frame_status_map") or {}
        frame_tags_map = rec.get("frame_tags_map") or {}

        frame_files = clips[i].get("frame_files") or []
        if not frame_files:
            frame_files = [f"frame_{k:06d}.jpg" for k in range(len(clips[i].get("frames", [])))]

        usable_frames = [f for f in frame_files if frame_status_map.get(f, "usable") == "usable"]

        # needs triage if a frame missing status key
        needs_triage = any(f not in frame_status_map for f in frame_files)

        # needs tags if any usable frame has empty tags
        needs_tags = any(len(frame_tags_map.get(f, [])) == 0 for f in usable_frames)

        # needs boxes if any usable frame has tags but has zero boxes
        boxes = rec.get("boxes", [])

        def frame_has_boxes(ff):
            return any(b.get("frame") == ff for b in boxes)

        needs_boxes = any((len(frame_tags_map.get(f, [])) > 0 and not frame_has_boxes(f)) for f in usable_frames)

        has_violations = any(len(frame_tags_map.get(f, [])) > 0 for f in usable_frames)

        complete = (not needs_tags) and (not needs_boxes)

        if mode == "All":
            return True
        if mode == "Needs Triage":
            return needs_triage
        if mode == "Needs Tags":
            return needs_tags
        if mode == "Needs Boxes":
            return needs_boxes
        if mode == "Complete":
            return complete
        if mode == "Has Violations":
            return has_violations
        return True

    i = st.session_state.clip_idx
    step = 1 if delta >= 0 else -1
    for _ in range(n):
        i = i + step
        if i < 0 or i >= n:
            break
        if ok(i):
            st.session_state.clip_idx = i
            load_clip_into_ui()
            return



def handle_jump():
    j = int(st.session_state.jump_to)
    clips = st.session_state.clips
    if not clips:
        return
    st.session_state.clip_idx = max(0, min(j, len(clips) - 1))
    load_clip_into_ui()

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def shapes_to_boxes(shapes, img_w: int, img_h: int, key_frame_name: str, prev_boxes=None, active_class=None):
    """
    Convert Plotly rect shapes to normalized YOLO center coords.
    Preserve labels by index; any NEW shape gets active_class.
    """
    prev_boxes = prev_boxes or []
    out = []
    if not shapes:
        return out

    for i, s in enumerate(shapes):
        x0 = float(getattr(s, "x0", 0))
        x1 = float(getattr(s, "x1", 0))
        y0 = float(getattr(s, "y0", 0))
        y1 = float(getattr(s, "y1", 0))

        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)

        w = right - left
        h = bottom - top
        if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
            continue

        cx = (left + w / 2.0) / img_w
        cy = (top + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        if i < len(prev_boxes):
            label = prev_boxes[i].get("label", BOX_CLASSES[0])
        else:
            label = active_class or BOX_CLASSES[0]

        out.append({"frame": key_frame_name, "label": label, "x": cx, "y": cy, "w": nw, "h": nh})

    return out


def make_draw_figure(img: Image.Image, initial_boxes, img_w: int, img_h: int, active_class: str):
    """
    Plotly figure where user can draw rectangles.
    We color shapes based on the box 'label'.
    """
    img_b64 = pil_to_base64_png(img)
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{img_b64}",
            x=0, y=0,
            sizex=img_w, sizey=img_h,
            xref="x", yref="y",
            sizing="stretch",
            layer="below",
        )
    )

    fig.update_xaxes(range=[0, img_w], visible=False, constrain="domain")
    fig.update_yaxes(range=[img_h, 0], visible=False, scaleanchor="x")

    shapes = []
    for b in initial_boxes or []:
        cx, cy, bw, bh = b["x"], b["y"], b["w"], b["h"]
        wpx = bw * img_w
        hpx = bh * img_h
        left = cx * img_w - wpx / 2.0
        top = cy * img_h - hpx / 2.0
        right = left + wpx
        bottom = top + hpx

        label = b.get("label", BOX_CLASSES[0])
        fill = BOX_CLASS_STYLE.get(label, "rgba(255,0,0,0.15)")

        shapes.append(
            dict(
                type="rect",
                x0=left, y0=top, x1=right, y1=bottom,
                line=dict(width=2),
                fillcolor=fill,
            )
        )

    fig.update_layout(
        shapes=shapes,
        dragmode="drawrect",
        newshape=dict(
            line=dict(width=2),
            fillcolor=BOX_CLASS_STYLE.get(active_class, "rgba(255,0,0,0.15)")
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=min(900, max(300, int(img_h * 0.9))),
        modebar_add=["drawrect", "eraseshape"],
    )
    return fig



def ui_selector_page():
    st.title("Traffic Violation Labeler — Phase-1")
    st.subheader("Project / Video selector")

    with st.expander("📌 How to use this labelling tool (quick workflow)", expanded=True):
        st.markdown(
            """
    **Step 1 — Select video + annotator**
    - Choose the video folder
    - Enter your name (used in Mongo records)

    **Step 2 — Start labeling**
    - Add **Violation Tags** (multi-label) for the clip (left sidebar)
    - Choose **Key frame** (slider)
    - Choose a **Box class**, then **draw boxes**
    - Edit labels if needed and hit **Save** or **Done & Next**
            """
        )


    db = get_db()
    ensure_indexes(db)

    videos = list_videos(db)
    if not videos:
        st.info("No videos found in MongoDB. Ingest first.")
        return

    video = st.selectbox("Pick a video folder", options=videos)
    st.text_input("Annotator name", value=st.session_state.annotator, key="annotator")

    if st.button("Load project stats"):
        load_project(video)

    if st.session_state.video_folder == video and st.session_state.clips:
        stats = get_video_stats(db, video)
        ann_map = st.session_state.ann_map
        clips = st.session_state.clips

        counts = compute_progress_counts(clips, ann_map)
        last_idx = get_last_labeled_clip_idx(clips, ann_map)

        st.markdown("### Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total clips", stats["total_clips"])
        c2.metric("Done", counts["done"])
        c3.metric("Skipped", counts["skipped"])
        c4.metric("Unclear", counts["unclear"])

        st.write(f"**Last labeled clip index:** {last_idx if last_idx is not None else '—'}")
        st.write(f"**Resume cursor (clip_idx):** {stats['cursor_clip_idx']}")

        if st.button("Start / Resume labeling"):
            st.session_state.page = "labeling"
            st.rerun()


def ui_keyboard_shortcuts():
    """
    Keyboard shortcuts via streamlit-keyup.
    Frame-level:
      A = prev clip
      D = next clip
      U = mark current frame unclear
      X = discard current frame
      C = usable current frame
      0 = clear tags for current frame
      1-4 = toggle tags for current frame
      S = save
    """
    try:
        from streamlit_keyup import st_keyup  # type: ignore
    except Exception:
        st.caption("Tip: install `streamlit-keyup` to enable keyboard shortcuts.")
        return

    key = st_keyup("Keyboard", key="__keyup__", label_visibility="collapsed")
    if not key:
        return

    key = str(key).strip().upper()
    if key == st.session_state.get("keyboard_last", ""):
        return
    st.session_state.keyboard_last = key

    # Need these to exist
    clip = current_clip()
    frame_files = clip.get("frame_files") or []
    if not frame_files:
        frame_files = [f"frame_{i:06d}.jpg" for i in range(len(clip.get("frames", [])))]

    st.session_state.frame_idx = max(0, min(int(st.session_state.frame_idx), len(frame_files) - 1))
    frame_name = frame_files[st.session_state.frame_idx]

    # Ensure maps exist
    st.session_state.frame_status_map.setdefault(frame_name, "usable")
    st.session_state.frame_tags_map.setdefault(frame_name, [])

    def toggle_frame_tag(tag: str):
        tags = set(st.session_state.frame_tags_map.get(frame_name, []))
        if tag in tags:
            tags.remove(tag)
        else:
            tags.add(tag)
        st.session_state.frame_tags_map[frame_name] = sorted(tags)
        autosave_annotation()

    if key == "A":
        move_next_filtered(-1)
        st.rerun()
    elif key == "D":
        move_next_filtered(1)
        st.rerun()
    elif key == "C":  # usable
        st.session_state.frame_status_map[frame_name] = "usable"
        autosave_annotation()
        st.rerun()
    elif key == "X":  # discard
        st.session_state.frame_status_map[frame_name] = "discarded"
        st.session_state.frame_tags_map[frame_name] = []
        st.session_state.canvas_boxes = []
        autosave_annotation()
        st.rerun()
    elif key == "U":  # unclear
        st.session_state.frame_status_map[frame_name] = "unclear"
        autosave_annotation()
        st.rerun()
    elif key == "0":
        st.session_state.frame_tags_map[frame_name] = []
        autosave_annotation()
        st.rerun()
    elif key == "1":
        toggle_frame_tag("no_helmet")
        st.rerun()
    elif key == "2":
        toggle_frame_tag("triple_riding")
        st.rerun()
    elif key == "3":
        toggle_frame_tag("signal_jump")
        st.rerun()
    elif key == "4":
        toggle_frame_tag("phone_driving")
        st.rerun()
    elif key == "S":
        autosave_annotation()
        st.toast("Saved", icon="✅")



def ui_labeling_page():
    # -----------------------------
    # Guard + basic vars
    # -----------------------------
    if not st.session_state.video_folder:
        st.session_state.page = "selector"
        st.rerun()

    video = st.session_state.video_folder
    clips = st.session_state.clips or []
    ann_map = st.session_state.ann_map or {}
    idx = int(st.session_state.clip_idx)
    n = len(clips)

    if n == 0:
        st.error("No clips loaded. Go back and load a project.")
        return

    idx = max(0, min(idx, n - 1))
    st.session_state.clip_idx = idx

    counts = compute_progress_counts(clips, ann_map)
    remaining = n - (counts.get("done", 0) + counts.get("skipped", 0) + counts.get("unclear", 0))
    done_total = counts.get("done", 0) + counts.get("skipped", 0) + counts.get("unclear", 0)
    progress = 0.0 if n == 0 else done_total / n

    st.title("Labeling workspace")
    st.write(f"**Video:** `{video}`  |  **Clip:** `{idx}/{n-1}`  |  **Progress:** {done_total}/{n}")
    st.progress(progress)

    ui_keyboard_shortcuts()

    # -----------------------------
    # Clip docs + frame lists
    # -----------------------------
    clip = clips[idx]
    frame_files = clip.get("frame_files") or []
    frame_ids = clip.get("frames") or []

    if not frame_files and frame_ids:
        frame_files = [f"frame_{i:06d}.jpg" for i in range(len(frame_ids))]

    if (not frame_ids) or (not frame_files) or (len(frame_ids) != len(frame_files)):
        st.error(
            "Clip document is missing expected fields. Expected: "
            "frame_files(list[str]) and frames(list[gridfs_id]) of same length."
        )
        st.stop()

    # Ensure frame_idx is valid
    st.session_state.frame_idx = max(0, min(int(st.session_state.frame_idx), len(frame_files) - 1))

    # Current working frame
    frame_name = frame_files[st.session_state.frame_idx]
    frame_id = frame_ids[st.session_state.frame_idx]

    # -----------------------------
    # Sidebar: Triage + Frame tags + Tools
    # -----------------------------
    with st.sidebar:
        st.header("Stage A — Triage + Tags (Frame-level)")
        st.caption("1) Mark frame usable/discarded/unclear")
        st.caption("2) Add violation tags for this frame (if any)")
        st.caption("3) Only if needed → draw boxes for this frame")

        # Defensive: ensure maps exist
        if "frame_status_map" not in st.session_state or not isinstance(st.session_state.frame_status_map, dict):
            st.session_state.frame_status_map = {}
        if "frame_tags_map" not in st.session_state or not isinstance(st.session_state.frame_tags_map, dict):
            st.session_state.frame_tags_map = {}

        # Ensure keys exist for all frames
        for f in frame_files:
            st.session_state.frame_status_map.setdefault(f, "usable")
            st.session_state.frame_tags_map.setdefault(f, [])

        st.subheader("Frame usability")
        cA1, cA2, cA3 = st.columns(3)
        if cA1.button("✅ Usable"):
            st.session_state.frame_status_map[frame_name] = "usable"
            autosave_annotation()
            st.rerun()
        if cA2.button("❌ Discard"):
            st.session_state.frame_status_map[frame_name] = "discarded"
            st.session_state.frame_tags_map[frame_name] = []
            st.session_state.canvas_boxes = []
            autosave_annotation()
            st.rerun()
        if cA3.button("🤷 Unclear"):
            st.session_state.frame_status_map[frame_name] = "unclear"
            autosave_annotation()
            st.rerun()

        st.write(f"Current: **{st.session_state.frame_status_map.get(frame_name, 'usable')}**")

        st.subheader("Violation tags (this frame)")

        def _on_frame_tag_change():
            # No need to call init_session() here; it can reset keys unexpectedly
            tags = []
            for t in VIOLATION_TAGS:
                if st.session_state.get(f"ftag_{t}", False):
                    tags.append(t)
            st.session_state.frame_tags_map[frame_name] = tags
            autosave_annotation()

        tags_set = set(st.session_state.frame_tags_map.get(frame_name, []))
        for t in VIOLATION_TAGS:
            st.checkbox(t, key=f"ftag_{t}", value=(t in tags_set), on_change=_on_frame_tag_change)

        st.divider()

        st.subheader("Tools")
        if st.button("Copy boxes from previous clip"):
            copy_boxes_from_previous_clip()
            st.rerun()
        if st.button("Clear boxes (this frame)"):
            clear_boxes_current_keyframe()
            st.rerun()
        if st.button("Propagate boxes to all frames (optional)"):
            propagate_boxes_to_all_frames_in_clip()
            st.toast("Propagated", icon="📌")
            st.rerun()

        st.number_input("Jump to clip #", min_value=0, max_value=max(0, n - 1), key="jump_to")
        if st.button("Jump"):
            handle_jump()
            st.rerun()

        st.selectbox("Filter mode", FILTER_MODES, key="filter_mode", on_change=autosave_state_only)

        st.divider()
        st.markdown("### Progress")
        st.write(f"Done: **{counts.get('done', 0)}**")
        st.write(f"Skipped: **{counts.get('skipped', 0)}**")
        st.write(f"Unclear: **{counts.get('unclear', 0)}**")
        st.write(f"Remaining: **{remaining}**")

        st.caption("Shortcuts: A prev, D next, U unclear, S save")

    # -----------------------------
    # Main: Frame thumbnails + frame selector
    # -----------------------------
    st.subheader(f"Clip {clip['clip_id']} — {len(frame_files)} frames")
    cols = st.columns(len(frame_files))
    for j, fname in enumerate(frame_files):
        thumb = load_frame_image_cached(video_id=video, frame_id=frame_ids[j], max_w=220)
        with cols[j]:
            st.image(thumb, caption=..., width="stretch")


    k = st.slider(
        "Select working frame (for triage/tags/boxes)",
        min_value=0,
        max_value=len(frame_files) - 1,
        value=int(st.session_state.frame_idx),
    )
    if k != st.session_state.frame_idx:
        st.session_state.frame_idx = k
        refresh_boxes_for_selected_frame()
        autosave_state_only()
        st.rerun()

    # Update working frame after slider
    frame_name = frame_files[st.session_state.frame_idx]
    frame_id = frame_ids[st.session_state.frame_idx]

    # -----------------------------
    # Main: Draw boxes UI
    # -----------------------------
    img = load_frame_image_cached(video_id=video, frame_id=frame_id, max_w=1100)
    img_w, img_h = img.size

    st.markdown("### Draw boxes (current frame)")
    st.caption("Boxes are saved as normalized YOLO-style center coords: x,y,w,h (0–1).")
    st.caption("Tip: draw rectangles. Use eraser tool to delete.")

    frame_status = st.session_state.frame_status_map.get(frame_name, "usable")
    frame_tags = st.session_state.frame_tags_map.get(frame_name, [])

    if frame_status != "usable":
        st.info("This frame is not marked usable. Mark it usable to draw boxes.")
        st.stop()

    if len(frame_tags) == 0:
        st.info(
            "No violation tags for this frame. If this is clean, leave tags empty. "
            "If violation exists, add tags first."
        )
        # Not stopping—allow user to draw boxes if they want.

    # Active class for new boxes
    st.markdown("#### Active Box Class (new boxes will use this label)")
    if "active_box_class" not in st.session_state:
        st.session_state.active_box_class = BOX_CLASSES[0]

    selected = st.selectbox(
        "Box class",
        BOX_CLASSES,
        index=BOX_CLASSES.index(st.session_state.active_box_class)
        if st.session_state.active_box_class in BOX_CLASSES
        else 0,
        key="active_box_class_select",
    )
    st.session_state.active_box_class = selected

    # Build draw figure (always rebuild from current boxes)
    fig = make_draw_figure(
        img=img,
        initial_boxes=st.session_state.canvas_boxes,
        img_w=img_w,
        img_h=img_h,
        active_class=st.session_state.active_box_class,
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})

    if frame_tags and not st.session_state.canvas_boxes:
        st.info("This frame has violation tags. Draw at least one relevant box (if localization is needed).")
    if st.session_state.canvas_boxes and len(frame_tags) == 0:
        st.info("You drew boxes but this frame has no violation tags. Add matching frame-level tags in the sidebar.")

    # Convert shapes → boxes and autosave if changed
    shapes = getattr(fig.layout, "shapes", []) or []
    new_boxes = shapes_to_boxes(
        shapes=shapes,
        img_w=img_w,
        img_h=img_h,
        key_frame_name=frame_name,  # function param name kept as-is
        prev_boxes=st.session_state.canvas_boxes,
        active_class=st.session_state.active_box_class,
    )

    new_hash = json.dumps(new_boxes, sort_keys=True)
    if new_hash != st.session_state.last_canvas_hash:
        st.session_state.canvas_boxes = new_boxes
        st.session_state.last_canvas_hash = new_hash
        autosave_annotation()

    # -----------------------------
    # Box list (editable labels)
    # -----------------------------
    st.markdown("### Box list (editable labels)")
    if not st.session_state.canvas_boxes:
        st.info("No boxes for this frame yet.")
    else:
        for i_b, b in enumerate(st.session_state.canvas_boxes):
            c1, c2, c3 = st.columns([2, 2, 6])
            with c1:
                st.write(f"Box {i_b+1}")
            with c2:
                new_label = st.selectbox(
                    "Class",
                    BOX_CLASSES,
                    index=BOX_CLASSES.index(b["label"]) if b.get("label") in BOX_CLASSES else 0,
                    key=f"boxlabel_{clip['clip_id']}_{frame_name}_{i_b}",
                )
                if new_label != b.get("label"):
                    st.session_state.canvas_boxes[i_b]["label"] = new_label
                    autosave_annotation()
                    st.rerun()
            with c3:
                st.write(
                    f"{b.get('frame', frame_name)} | "
                    f"x={b.get('x', 0):.3f}, y={b.get('y', 0):.3f}, "
                    f"w={b.get('w', 0):.3f}, h={b.get('h', 0):.3f}"
                )

    st.divider()
    st.caption("Tip: Triage + tags first. Boxes only for frames that actually need localization.")


def main():
    st.set_page_config(page_title="Traffic Violation Labeler (Phase-1)", layout="wide")
    init_session()

    if st.session_state.page == "selector":
        ui_selector_page()
    else:
        ui_labeling_page()


if __name__ == "__main__":
    main()
