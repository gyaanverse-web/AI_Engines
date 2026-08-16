import json
import os
from dotenv import load_dotenv

load_dotenv()
from testing_engine import qdrant_client, QDRANT_COLLECTION_NAME

COLLECTION_NAME = QDRANT_COLLECTION_NAME or "ncert_local_test"
DB_PATH = os.path.join(os.path.dirname(__file__), "qdrant_local_db")

def get_dir_size(path):
    total_size = 0
    if os.path.exists(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    return total_size

def view_database():
    print(f"Connecting to Local Qdrant Database...")
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"❌ Collection '{COLLECTION_NAME}' does not exist yet.")
            return

        # Get total count of questions saved
        count_result = qdrant_client.count(collection_name=COLLECTION_NAME)
        
        # Get disk size
        size_bytes = get_dir_size(DB_PATH)
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n✅ Total Questions Saved in '{COLLECTION_NAME}': {count_result.count}")
        print(f"💾 Database Size on Disk: {size_mb:.2f} MB\n")
        
        if count_result.count == 0:
            print("Database is currently empty. Run index_ncert_qdrant.py first.")
            return
            
        print("Showing the first 5 records (Payload only)...")
        print("=" * 80)
        
        # Scroll lets us view records sequentially without searching by a vector
        records, next_page_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False # False because we don't want to flood the screen with 4096 numbers
        )
        
        for i, record in enumerate(records):
            print(f"\n🔹 RECORD {i+1} | ID: {record.id}")
            print("-" * 80)
            
            # Print payload in a nice readable JSON format
            print(json.dumps(record.payload, indent=4))
            print("=" * 80)

    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    view_database()
