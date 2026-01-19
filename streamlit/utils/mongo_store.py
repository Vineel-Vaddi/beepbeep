import os
from typing import Optional, Dict, Any
from pymongo import MongoClient, ASCENDING
import gridfs

def get_client(mongo_uri: str) -> MongoClient:
    return MongoClient(mongo_uri)

def get_db(client: MongoClient, db_name: str):
    return client[db_name]

def get_fs(db):
    return gridfs.GridFS(db, collection="frames_fs")

def ensure_indexes(db):
    db.videos.create_index([("video_id", ASCENDING)], unique=True)
    db.clips.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)
    db.annotations.create_index([("video_id", ASCENDING), ("clip_id", ASCENDING)], unique=True)
    db.state.create_index([("video_id", ASCENDING)], unique=True)

    # For GridFS metadata queries (optional helpful indexes)
    db.frames_fs.files.create_index([("metadata.video_id", ASCENDING)])
    db.frames_fs.files.create_index([("metadata.video_id", ASCENDING), ("metadata.filename", ASCENDING)], unique=True)

def put_frame_bytes(fs, jpg_bytes: bytes, filename: str, metadata: Dict[str, Any]) -> str:
    """
    Returns GridFS file_id as string.
    """
    file_id = fs.put(jpg_bytes, filename=filename, metadata=metadata)
    return str(file_id)

def find_frame_file(fs, db, video_id: str, filename: str) -> Optional[str]:
    """
    Check if frame exists in GridFS already.
    """
    doc = db.frames_fs.files.find_one({"metadata.video_id": video_id, "metadata.filename": filename}, {"_id": 1})
    return str(doc["_id"]) if doc else None
