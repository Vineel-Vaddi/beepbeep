from io import BytesIO
from pathlib import Path
from typing import Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service(service_account_info: dict):
    """
    Build Google Drive service from service account info dict.
    """
    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def create_folder(service, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
    """
    Create a folder in Google Drive and return its folder ID.
    """
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    }

    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]

    folder = service.files().create(
        body=file_metadata,
        fields="id"
    ).execute()

    return folder["id"]


def upload_file_to_drive(
    service,
    file_path: str,
    parent_folder_id: Optional[str] = None
) -> str:
    """
    Upload a local file to Google Drive and return file ID.
    """
    file_path = Path(file_path)

    file_metadata = {"name": file_path.name}
    if parent_folder_id:
        file_metadata["parents"] = [parent_folder_id]

    with open(file_path, "rb") as f:
        file_bytes = BytesIO(f.read())

    media = MediaIoBaseUpload(file_bytes, mimetype="image/jpeg", resumable=True)

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    return uploaded_file["id"]


def upload_frames_to_drive(
    frame_paths: list[str],
    service_account_info: dict,
    root_folder_id: str,
    upload_folder_name: str
) -> dict:
    """
    Create a subfolder inside the root Drive folder and upload all frames there.

    Returns:
        {
            "folder_id": ...,
            "uploaded_count": ...,
            "file_ids": [...]
        }
    """
    service = get_drive_service(service_account_info)

    subfolder_id = create_folder(
        service=service,
        folder_name=upload_folder_name,
        parent_folder_id=root_folder_id
    )

    uploaded_file_ids = []

    for frame_path in frame_paths:
        file_id = upload_file_to_drive(
            service=service,
            file_path=frame_path,
            parent_folder_id=subfolder_id
        )
        uploaded_file_ids.append(file_id)

    return {
        "folder_id": subfolder_id,
        "uploaded_count": len(uploaded_file_ids),
        "file_ids": uploaded_file_ids
    }