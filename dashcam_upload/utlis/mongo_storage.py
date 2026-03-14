from datetime import datetime
from typing import List, Optional

from pymongo import MongoClient
from gridfs import GridFS


def get_mongo_client(mongo_uri: str) -> MongoClient:
    return MongoClient(mongo_uri)


def get_database(client: MongoClient, db_name: str = "dashcam_app"):
    return client[db_name]


def create_upload_record(
    mongo_uri: str,
    video_name: str,
    total_extracted: int,
    duplicates_removed: int,
    final_saved: int,
    status: str = "completed",
    db_name: str = "dashcam_app"
) -> str:
    """
    Insert one upload metadata record and return inserted id as string.
    """
    client = get_mongo_client(mongo_uri)
    db = get_database(client, db_name)

    doc = {
        "video_name": video_name,
        "status": status,
        "total_extracted": total_extracted,
        "duplicates_removed": duplicates_removed,
        "final_saved": final_saved,
        "created_at": datetime.utcnow()
    }

    result = db["uploads_metadata"].insert_one(doc)
    client.close()
    return str(result.inserted_id)


def save_frame_to_gridfs(
    mongo_uri: str,
    file_path: str,
    video_name: str,
    upload_id: Optional[str] = None,
    db_name: str = "dashcam_app"
) -> str:
    """
    Save one frame file into GridFS and return file id as string.
    """
    client = get_mongo_client(mongo_uri)
    db = get_database(client, db_name)
    fs = GridFS(db)

    with open(file_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=file_path.split("/")[-1],
            content_type="image/jpeg",
            video_name=video_name,
            upload_id=upload_id,
            created_at=datetime.utcnow()
        )

    client.close()
    return str(file_id)


def save_multiple_frames_to_gridfs(
    mongo_uri: str,
    frame_paths: List[str],
    video_name: str,
    upload_id: Optional[str] = None,
    db_name: str = "dashcam_app"
) -> List[str]:
    """
    Save multiple frames into GridFS and return list of file ids.
    """
    file_ids = []

    client = get_mongo_client(mongo_uri)
    db = get_database(client, db_name)
    fs = GridFS(db)

    for file_path in frame_paths:
        with open(file_path, "rb") as f:
            file_id = fs.put(
                f,
                filename=file_path.split("/")[-1],
                content_type="image/jpeg",
                video_name=video_name,
                upload_id=upload_id,
                created_at=datetime.utcnow()
            )
            file_ids.append(str(file_id))

    client.close()
    return file_ids