import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from utils.frame_extractor import extract_frames
from utils.deduplicate import remove_duplicate_frames
from utils.mongo_storage import (
    create_upload_record,
    save_multiple_frames_to_gridfs,
)

st.set_page_config(page_title="Dashcam Upload", page_icon="🚗", layout="centered")

st.title("🚗 Dashcam Video Uploader")
st.write("Upload a dashcam video, extract frames, remove duplicates, and save final frames to MongoDB.")

# -----------------------------
# Secrets
# -----------------------------
try:
    MONGO_URI = st.secrets["mongo"]["uri"]
except Exception:
    st.error("MongoDB connection string is missing in Streamlit secrets.")
    st.stop()

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload dashcam video",
    type=["mp4", "mov", "avi", "mkv"]
)

interval_seconds = st.number_input(
    "Extract 1 frame every N seconds",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

process_btn = st.button("Process and Save")

# -----------------------------
# Main flow
# -----------------------------
if process_btn:
    if not uploaded_file:
        st.warning("Please upload a video first.")
        st.stop()

    temp_dir = tempfile.mkdtemp(prefix="dashcam_")
    video_path = os.path.join(temp_dir, uploaded_file.name)
    frames_dir = os.path.join(temp_dir, "frames")

    try:
        # Save uploaded video temporarily
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Video uploaded. Extracting frames...")

        # Step 1: Extract frames
        frame_paths = extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            interval_seconds=int(interval_seconds)
        )
        total_extracted = len(frame_paths)

        if total_extracted == 0:
            st.warning("No frames could be extracted from this video.")
            st.stop()

        st.info(f"Extracted {total_extracted} frames. Removing duplicates...")

        # Step 2: Remove duplicates
        unique_frames, removed_frames = remove_duplicate_frames(frame_paths)
        removed_count = len(removed_frames)
        final_count = len(unique_frames)

        if final_count == 0:
            st.warning("No frames remained after duplicate removal.")
            st.stop()

        st.info(f"Removed {removed_count} duplicates. Saving {final_count} frames to MongoDB...")

        # Step 3: Create metadata record
        upload_id = create_upload_record(
            mongo_uri=MONGO_URI,
            video_name=uploaded_file.name,
            total_extracted=total_extracted,
            duplicates_removed=removed_count,
            final_saved=final_count,
            status="completed"
        )

        # Step 4: Save frames to GridFS
        frame_file_ids = save_multiple_frames_to_gridfs(
            mongo_uri=MONGO_URI,
            frame_paths=unique_frames,
            video_name=uploaded_file.name,
            upload_id=upload_id
        )

        # Step 5: Success
        st.success("Upload successful!")
        st.write(f"**Video name:** {uploaded_file.name}")
        st.write(f"**Upload ID:** `{upload_id}`")
        st.write(f"**Total extracted frames:** {total_extracted}")
        st.write(f"**Duplicate frames removed:** {removed_count}")
        st.write(f"**Final frames saved:** {len(frame_file_ids)}")

        with st.expander("Saved frame file IDs"):
            for fid in frame_file_ids[:20]:
                st.code(fid)
            if len(frame_file_ids) > 20:
                st.caption(f"...and {len(frame_file_ids) - 20} more")

    except Exception as e:
        st.error(f"Error: {str(e)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)