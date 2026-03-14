from pymongo import MongoClient
from pymongo.errors import PyMongoError

MONGO_URI = "mongodb+srv://techwithaibuddies_db_user:FcbzZS0tRfMDEBDO@cluster0.naa9gp.mongodb.net/"

def main():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ping")
        print("Connected successfully.\n")

        db = client["dashcam_app"]
        print("Database:", db.name)

        collections = db.list_collection_names()
        print("Collections:", collections)

        for coll_name in collections:
            coll = db[coll_name]
            print(f"\nCollection: {coll_name}")
            print("Document count:", coll.count_documents({}))

            sample = coll.find_one()
            print("Sample document:")
            print(sample)

    except PyMongoError as e:
        print("MongoDB error:", e)
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == "__main__":
    main()