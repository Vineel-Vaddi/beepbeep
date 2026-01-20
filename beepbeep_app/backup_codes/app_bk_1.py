import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils.frames import (
    list_video_folders,
    build_or_load_index,
    build_clips_from_index,
    get_clip_frames,
    load_image_cached,
)
from utils.storage import (
    ensure_video_annotation_dir,
    load_latest_annotations_map,
    load_state,
    save_state,
    append_annotation_record,
    compute_progress_counts,
    get_last_labeled_clip_idx,
)

# -----------------------------
# Fixed label sets (Phase-1)
# -----------------------------
VIOLATION_TAGS = ["no_helmet", "triple_riding", "signal_jump", "phone_driving"]
BOX_CLASSES = ["plate", "bike", "rider_head", "rider_hand", "traffic_signal"]

DEFAULT_DATASET_ROOT = "frames_out"
DEFAULT_ANNOTATOR = "user1"

FILTER_MODES = ["All", "Unlabeled", "Done", "Skipped", "Unclear", "Positive-only"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_session():
    if "page" not in st.session_state:
        st.session_state.page = "selector"  # selector | labeling
    if "dataset_root" not in st.session_state:
        st.session_state.dataset_root = DEFAULT_DATASET_ROOT
    if "video_folder" not in st.session_state:
        st.session_state.video_folder = None
    if "annotator" not in st.session_state:
        st.session_state.annotator = DEFAULT_ANNOTATOR

    # Index / clips cache in session
    if "index_data" not in st.session_state:
        st.session_state.index_data = None
    if "clips" not in st.session_state:
        st.session_state.clips = None

    # Annotations map (latest per clip_id)
    if "ann_map" not in st.session_state:
        st.session_state.ann_map = {}

    # Cursor
    if "clip_idx" not in st.session_state:
        st.session_state.clip_idx = 0

    # UI state
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "All"
    if "jump_to" not in st.session_state:
        st.session_state.jump_to = 0
    if "key_frame_idx" not in st.session_state:
        st.session_state.key_frame_idx = 0  # within clip

    # Canvas persistence
    if "canvas_boxes" not in st.session_state:
        st.session_state.canvas_boxes = []  # list of dicts in schema format
    if "last_canvas_hash" not in st.session_state:
        st.session_state.last_canvas_hash = ""

    if "tags" not in st.session_state:
        st.session_state.tags = []
    if "review_status" not in st.session_state:
        st.session_state.review_status = "unlabeled"  # done|skipped|unclear|unlabeled

    if "keyboard_last" not in st.session_state:
        st.session_state.keyboard_last = ""


def get_video_paths(dataset_root: str, video_folder: str) -> Dict[str, str]:
    frames_dir = os.path.join(dataset_root, video_folder)
    ann_dir = ensure_video_annotation_dir(video_folder)
    return {"frames_dir": frames_dir, "ann_dir": ann_dir}


def load_project(video_folder: str):
    dataset_root = st.session_state.dataset_root
    paths = get_video_paths(dataset_root, video_folder)

    # Build/load dedupe index
    index_data = build_or_load_index(
        frames_dir=paths["frames_dir"],
        ann_dir=paths["ann_dir"],
        dhash_threshold=6,
    )
    clips = build_clips_from_index(index_data, clip_len=4)

    # Load latest annotations + state
    ann_map = load_latest_annotations_map(paths["ann_dir"])
    state = load_state(paths["ann_dir"])

    # Restore cursor
    clip_idx = state.get("cursor_clip_idx", 0)
    clip_idx = max(0, min(clip_idx, max(0, len(clips) - 1)))

    st.session_state.video_folder = video_folder
    st.session_state.index_data = index_data
    st.session_state.clips = clips
    st.session_state.ann_map = ann_map
    st.session_state.clip_idx = clip_idx
    st.session_state.filter_mode = state.get("filter_mode", "All")

    # Load current clip into UI
    load_clip_into_ui()


def load_clip_into_ui():
    """Populate UI fields (tags/status/boxes) from ann_map for current clip."""
    video_folder = st.session_state.video_folder
    if not video_folder:
        return

    clips = st.session_state.clips
    if not clips:
        return

    clip_idx = st.session_state.clip_idx
    clip_idx = max(0, min(clip_idx, len(clips) - 1))
    clip = clips[clip_idx]
    clip_id = clip["clip_id"]

    rec = st.session_state.ann_map.get(clip_id)

    # default values
    st.session_state.tags = []
    st.session_state.review_status = "unlabeled"
    st.session_state.canvas_boxes = []
    st.session_state.key_frame_idx = min(1, len(clip["frames"]) - 1) if len(clip["frames"]) > 1 else 0

    if rec:
        st.session_state.tags = rec.get("tags", [])
        st.session_state.review_status = rec.get("review_status", "unlabeled")

        # Boxes: store only those for the currently selected key frame by default
        # We'll reload boxes for the selected key frame when key frame changes.
        # Here, just initialize from first frame in clip (or empty), then refresh.
        st.session_state.canvas_boxes = []
        st.session_state.key_frame_idx = rec.get("key_frame_idx", st.session_state.key_frame_idx)

    refresh_boxes_for_key_frame()
    autosave_state_only()


def refresh_boxes_for_key_frame():
    """Load boxes from ann_map for current clip & selected key frame into session_state.canvas_boxes."""
    clips = st.session_state.clips
    if not clips:
        return
    clip = clips[st.session_state.clip_idx]
    clip_id = clip["clip_id"]
    frames = clip["frames"]
    k = max(0, min(st.session_state.key_frame_idx, len(frames) - 1))
    key_frame = frames[k]

    rec = st.session_state.ann_map.get(clip_id)
    if not rec:
        st.session_state.canvas_boxes = []
        st.session_state.last_canvas_hash = ""
        return

    boxes = rec.get("boxes", [])
    boxes_k = [b for b in boxes if b.get("frame") == key_frame]
    st.session_state.canvas_boxes = boxes_k
    st.session_state.last_canvas_hash = json.dumps(boxes_k, sort_keys=True)


def current_clip() -> Dict[str, Any]:
    return st.session_state.clips[st.session_state.clip_idx]


def clip_record_from_ui() -> Dict[str, Any]:
    video_id = st.session_state.video_folder
    clip = current_clip()
    clip_id = clip["clip_id"]
    frames = clip["frames"]
    k = max(0, min(st.session_state.key_frame_idx, len(frames) - 1))
    key_frame = frames[k]

    # Merge: keep other frames' boxes from existing record (if any), but replace key_frame boxes from UI
    existing = st.session_state.ann_map.get(clip_id, {})
    existing_boxes = existing.get("boxes", [])
    other_boxes = [b for b in existing_boxes if b.get("frame") != key_frame]

    ui_boxes = st.session_state.canvas_boxes or []
    merged_boxes = other_boxes + ui_boxes

    rec = {
        "video_id": video_id,
        "clip_id": clip_id,
        "frames": frames,
        "tags": sorted(list(set(st.session_state.tags))),
        "boxes": merged_boxes,
        "review_status": st.session_state.review_status,
        "annotator": st.session_state.annotator,
        "updated_at": now_iso(),
        # helpful extra (not required but harmless)
        "key_frame_idx": k,
        "key_frame": key_frame,
    }
    return rec


def autosave_annotation():
    """Append record to JSONL and update ann_map (latest wins)."""
    if not st.session_state.video_folder:
        return

    paths = get_video_paths(st.session_state.dataset_root, st.session_state.video_folder)
    rec = clip_record_from_ui()

    append_annotation_record(paths["ann_dir"], rec)
    st.session_state.ann_map[rec["clip_id"]] = rec

    # Update state cursor
    autosave_state_only()


def autosave_state_only():
    if not st.session_state.video_folder:
        return
    paths = get_video_paths(st.session_state.dataset_root, st.session_state.video_folder)

    counts = compute_progress_counts(st.session_state.clips, st.session_state.ann_map)
    state = {
        "video_id": st.session_state.video_folder,
        "dataset_root": st.session_state.dataset_root,
        "cursor_clip_idx": int(st.session_state.clip_idx),
        "filter_mode": st.session_state.filter_mode,
        "updated_at": now_iso(),
        "counts": counts,
    }
    save_state(paths["ann_dir"], state)


def set_status(status: str):
    st.session_state.review_status = status
    autosave_annotation()


def clear_tags():
    st.session_state.tags = []
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
    frames = cur["frames"]
    k = max(0, min(st.session_state.key_frame_idx, len(frames) - 1))
    key_frame = frames[k]

    # Copy boxes from prev record's key_frame if present; else copy all boxes from prev and attach to current key frame
    prev_key = prev_rec.get("key_frame")
    prev_boxes = prev_rec.get("boxes", [])
    if prev_key:
        src = [b for b in prev_boxes if b.get("frame") == prev_key]
    else:
        src = prev_boxes[:]

    copied = []
    for b in src:
        copied.append(
            {
                "frame": key_frame,
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
    """Optional helper: duplicate current keyframe boxes to all frames in clip."""
    cur = current_clip()
    frames = cur["frames"]
    k = max(0, min(st.session_state.key_frame_idx, len(frames) - 1))
    key_frame = frames[k]
    ui_boxes = st.session_state.canvas_boxes or []

    # Create per-frame boxes
    all_boxes = []
    for f in frames:
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

    # Replace all frames boxes in record
    clip_id = cur["clip_id"]
    rec = st.session_state.ann_map.get(clip_id, {})
    rec_boxes_existing = rec.get("boxes", [])
    rec_boxes_nonclip = [b for b in rec_boxes_existing if b.get("frame") not in frames]
    st.session_state.ann_map[clip_id] = {**clip_record_from_ui(), "boxes": rec_boxes_nonclip + all_boxes}
    autosave_annotation()


def move_cursor(delta: int):
    clips = st.session_state.clips
    n = len(clips)
    st.session_state.clip_idx = max(0, min(st.session_state.clip_idx + delta, n - 1))
    load_clip_into_ui()


def move_next_filtered(delta: int):
    """Move to next/prev clip respecting filter mode."""
    mode = st.session_state.filter_mode
    clips = st.session_state.clips
    n = len(clips)

    def ok(i: int) -> bool:
        clip_id = clips[i]["clip_id"]
        rec = st.session_state.ann_map.get(clip_id)
        status = rec.get("review_status", "unlabeled") if rec else "unlabeled"
        tags = rec.get("tags", []) if rec else []
        if mode == "All":
            return True
        if mode == "Unlabeled":
            return status == "unlabeled"
        if mode == "Done":
            return status == "done"
        if mode == "Skipped":
            return status == "skipped"
        if mode == "Unclear":
            return status == "unclear"
        if mode == "Positive-only":
            return len(tags) > 0
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


def normalize_box_from_canvas(obj: Dict[str, Any], img_w: int, img_h: int) -> Optional[Dict[str, float]]:
    """
    Canvas rect object -> normalized YOLO-style center coords.
    st_canvas returns rect with left/top/width/height in pixels.
    Output: x,y,w,h in [0,1] where x,y are center.
    """
    if obj.get("type") != "rect":
        return None
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    w = float(obj.get("width", 0))
    h = float(obj.get("height", 0))
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None

    cx = (left + w / 2.0) / img_w
    cy = (top + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    # clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return {"x": cx, "y": cy, "w": nw, "h": nh}


def ui_selector_page():
    st.title("Traffic Violation Labeler — Phase-1")

    st.subheader("Project / Video selector")

    st.text_input("Dataset root", value=st.session_state.dataset_root, key="dataset_root")
    dataset_root = st.session_state.dataset_root

    if not os.path.isdir(dataset_root):
        st.warning(f"Folder not found: {dataset_root}")
        return

    videos = list_video_folders(dataset_root)
    if not videos:
        st.info("No video folders found under dataset root.")
        return

    video = st.selectbox("Pick a video folder", options=videos)
    st.text_input("Annotator name", value=st.session_state.annotator, key="annotator")

    if st.button("Load project stats"):
        load_project(video)

    if st.session_state.video_folder == video and st.session_state.index_data and st.session_state.clips:
        paths = get_video_paths(dataset_root, video)
        state = load_state(paths["ann_dir"])
        ann_map = st.session_state.ann_map
        clips = st.session_state.clips

        counts = compute_progress_counts(clips, ann_map)
        last_idx = get_last_labeled_clip_idx(clips, ann_map)

        total_frames = len(st.session_state.index_data["kept_frames"])
        st.markdown("### Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total (kept frames)", total_frames)
        c2.metric("Done", counts["done"])
        c3.metric("Skipped", counts["skipped"])
        c4.metric("Unclear", counts["unclear"])

        st.write(f"**Total clips:** {len(clips)}")
        st.write(f"**Last labeled clip index:** {last_idx if last_idx is not None else '—'}")
        st.write(f"**Resume cursor (clip_idx):** {state.get('cursor_clip_idx', 0)}")

        if st.button("Start / Resume labeling"):
            st.session_state.page = "labeling"
            st.rerun()


def ui_keyboard_shortcuts():
    """
    Keyboard shortcuts via streamlit-keyup (recommended).
    If not installed, app still works via buttons.
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
    if key == st.session_state.keyboard_last:
        return
    st.session_state.keyboard_last = key

    # Required mappings
    if key == "A":
        move_next_filtered(-1)
        st.rerun()
    elif key == "D":
        move_next_filtered(1)
        st.rerun()
    elif key == "0":
        clear_tags()
        st.rerun()
    elif key == "1":
        toggle_tag("no_helmet")
        st.rerun()
    elif key == "2":
        toggle_tag("triple_riding")
        st.rerun()
    elif key == "3":
        toggle_tag("signal_jump")
        st.rerun()
    elif key == "4":
        toggle_tag("phone_driving")
        st.rerun()
    elif key == "U":
        set_status("unclear")
        move_next_filtered(1)
        st.rerun()
    elif key == "S":
        autosave_annotation()
        st.toast("Saved", icon="✅")


def ui_labeling_page():
    if not st.session_state.video_folder:
        st.session_state.page = "selector"
        st.rerun()

    video = st.session_state.video_folder
    clips = st.session_state.clips
    ann_map = st.session_state.ann_map
    idx = st.session_state.clip_idx
    n = len(clips)

    # Top bar
    counts = compute_progress_counts(clips, ann_map)
    remaining = n - (counts["done"] + counts["skipped"] + counts["unclear"])
    done_total = counts["done"] + counts["skipped"] + counts["unclear"]
    progress = 0.0 if n == 0 else done_total / n

    st.title("Labeling workspace")
    st.write(f"**Video:** `{video}`  |  **Clip:** `{idx}/{n-1}`  |  **Progress:** {done_total}/{n}")
    st.progress(progress)

    # Keyboard shortcuts hook (must-have)
    ui_keyboard_shortcuts()

    # Sidebar
    with st.sidebar:
        st.header("Tags (clip-level)")
        # Render tag checkboxes with autosave on change
        def _on_tag_change():
            # build tags from checkbox states
            tags = []
            for t in VIOLATION_TAGS:
                if st.session_state.get(f"tag_{t}", False):
                    tags.append(t)
            st.session_state.tags = tags
            # If tags exist, don't force done; user chooses status. Just autosave.
            autosave_annotation()

        # Initialize checkbox values from session tags
        tags_set = set(st.session_state.tags)
        for t in VIOLATION_TAGS:
            st.checkbox(t, key=f"tag_{t}", value=(t in tags_set), on_change=_on_tag_change)

        st.divider()

        st.subheader("Quick buttons")
        colb1, colb2, colb3 = st.columns(3)
        if colb1.button("No violation (0)"):
            clear_tags()
            st.toast("Tags cleared", icon="🧹")
        if colb2.button("Unclear (U)"):
            set_status("unclear")
            move_next_filtered(1)
            st.rerun()
        if colb3.button("Save (S)"):
            autosave_annotation()
            st.toast("Saved", icon="✅")

        st.divider()

        st.subheader("Review status")
        st.write(f"Current: **{st.session_state.review_status}**")

        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Done & Next"):
            set_status("done")
            move_next_filtered(1)
            st.rerun()
        if col2.button("⏭ Skip & Next"):
            set_status("skipped")
            move_next_filtered(1)
            st.rerun()
        if col3.button("🤷 Unclear & Next"):
            set_status("unclear")
            move_next_filtered(1)
            st.rerun()

        col4, col5 = st.columns(2)
        if col4.button("↩ Prev (A)"):
            move_next_filtered(-1)
            st.rerun()
        if col5.button("⏩ Next (D)"):
            move_next_filtered(1)
            st.rerun()

        st.divider()

        st.subheader("Tools")
        if st.button("Copy boxes from previous"):
            copy_boxes_from_previous_clip()
            st.rerun()
        if st.button("Clear boxes (key frame)"):
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
        st.write(f"Done: **{counts['done']}**")
        st.write(f"Skipped: **{counts['skipped']}**")
        st.write(f"Unclear: **{counts['unclear']}**")
        st.write(f"Remaining: **{remaining}**")

        st.caption("Shortcuts: A prev, D next, 0 clear tags, 1-4 toggle tags, U unclear, S save")

    # Main area
    clip = clips[idx]
    frames = clip["frames"]

    # Mini strip / key frame selector
    st.subheader(f"Clip {clip['clip_id']} — {len(frames)} frames")
    cols = st.columns(len(frames))
    for j, f in enumerate(frames):
        img_path = os.path.join(st.session_state.dataset_root, video, f)
        img = load_image_cached(img_path, max_w=220)
        with cols[j]:
            st.image(img, caption=f"{j}: {f}", use_container_width=True)

    k = st.slider("Select key frame for boxes", min_value=0, max_value=len(frames) - 1, value=st.session_state.key_frame_idx)
    if k != st.session_state.key_frame_idx:
        st.session_state.key_frame_idx = k
        refresh_boxes_for_key_frame()
        autosave_state_only()

    key_frame = frames[st.session_state.key_frame_idx]
    key_path = os.path.join(st.session_state.dataset_root, video, key_frame)
    img = load_image_cached(key_path, max_w=1100)
    img_w, img_h = img.size

    st.markdown("### Draw boxes (key frame)")
    st.caption("Boxes are saved as normalized YOLO-style center coords: x,y,w,h (0–1).")

    # Build initial drawing objects from existing boxes (convert back to pixel rect)
    # NOTE: drawable-canvas expects pixel rect with left/top/width/height.
    initial_objects = []
    for b in st.session_state.canvas_boxes:
        # convert center-normalized -> pixel top-left
        cx, cy, bw, bh = b["x"], b["y"], b["w"], b["h"]
        wpx = bw * img_w
        hpx = bh * img_h
        left = cx * img_w - wpx / 2.0
        top = cy * img_h - hpx / 2.0
        initial_objects.append(
            {
                "type": "rect",
                "left": left,
                "top": top,
                "width": wpx,
                "height": hpx,
                "fill": "rgba(255, 0, 0, 0.15)",
                "stroke": "rgba(255, 0, 0, 0.9)",
                "strokeWidth": 2,
            }
        )

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.15)",
        stroke_width=2,
        stroke_color="rgba(255, 0, 0, 0.9)",
        background_image=img,
        update_streamlit=True,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        initial_drawing={"version": "4.4.0", "objects": initial_objects},
        key=f"canvas_{video}_{clip['clip_id']}_{key_frame}",
    )

    # Convert canvas rects -> normalized boxes, and preserve labels using a stable order
    # We keep labels in session_state.canvas_boxes (by index). If object count changes, default to "plate".
    new_boxes = []
    objects = (canvas_result.json_data or {}).get("objects", []) if canvas_result else []
    for i_obj, obj in enumerate(objects):
        nb = normalize_box_from_canvas(obj, img_w, img_h)
        if nb is None:
            continue
        label = BOX_CLASSES[0]
        if i_obj < len(st.session_state.canvas_boxes):
            label = st.session_state.canvas_boxes[i_obj].get("label", BOX_CLASSES[0])
        new_boxes.append({"frame": key_frame, "label": label, **nb})

    # Autosave on box change
    new_hash = json.dumps(new_boxes, sort_keys=True)
    if new_hash != st.session_state.last_canvas_hash:
        st.session_state.canvas_boxes = new_boxes
        st.session_state.last_canvas_hash = new_hash
        autosave_annotation()

    st.markdown("### Box list (editable labels)")
    if not st.session_state.canvas_boxes:
        st.info("No boxes for this key frame yet.")
    else:
        for i_b, b in enumerate(st.session_state.canvas_boxes):
            c1, c2, c3 = st.columns([2, 2, 6])
            with c1:
                st.write(f"Box {i_b+1}")
            with c2:
                new_label = st.selectbox(
                    "Class",
                    BOX_CLASSES,
                    index=BOX_CLASSES.index(b["label"]) if b["label"] in BOX_CLASSES else 0,
                    key=f"boxlabel_{clip['clip_id']}_{key_frame}_{i_b}",
                )
                if new_label != b["label"]:
                    st.session_state.canvas_boxes[i_b]["label"] = new_label
                    autosave_annotation()
                    st.rerun()
            with c3:
                st.write(f"{b['frame']} | x={b['x']:.3f}, y={b['y']:.3f}, w={b['w']:.3f}, h={b['h']:.3f}")

    # Pass-1 triage helpers
    st.divider()
    st.subheader("Pass-1 triage helpers")
    ctri1, ctri2, ctri3 = st.columns(3)
    if ctri1.button("0 — No violation (clear tags)"):
        clear_tags()
        st.toast("No violation set (tags cleared)", icon="✅")
    if ctri2.button("Mark Done (triage)"):
        # If no tags, done still allowed as "no violation" clip
        set_status("done")
        st.toast("Marked done", icon="✅")
    if ctri3.button("Skip (blur)"):
        set_status("skipped")
        st.toast("Skipped", icon="⏭")

    st.caption("Tip: For violations (any tag), do Pass-2: draw boxes on key frame (and optionally propagate).")


def main():
    st.set_page_config(page_title="Traffic Violation Labeler (Phase-1)", layout="wide")
    init_session()

    if st.session_state.page == "selector":
        ui_selector_page()
    else:
        ui_labeling_page()


if __name__ == "__main__":
    main()
