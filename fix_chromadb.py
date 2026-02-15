import os
import shutil
import chromadb
from chromadb.config import Settings
from pathlib import Path
import json

def backup_chroma_metadata():
    """Backup collection metadata before reset"""
    backup_dir = Path("./chroma_backup")
    backup_dir.mkdir(exist_ok=True)
    
    try:
        # Try to connect to existing DB to extract metadata
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections_info = []
        try:
            collections = client.list_collections()
            for collection in collections:
                collections_info.append({
                    'name': collection.name,
                    'metadata': collection.metadata,
                    'count': collection.count()
                })
        except Exception as e:
            print(f"Could not extract collection info: {e}")
        
        # Save collection metadata
        with open(backup_dir / "collections_metadata.json", "w") as f:
            json.dump(collections_info, f, indent=2)
        
        print(f"Backed up metadata for {len(collections_info)} collections")
        return collections_info
        
    except Exception as e:
        print(f"Could not backup metadata: {e}")
        return []

def reset_chromadb():
    """Reset ChromaDB by removing corrupted database files"""
    db_path = Path("./chroma_db")
    
    print("Backing up collection metadata...")
    collections_info = backup_chroma_metadata()
    
    if db_path.exists():
        print("Removing corrupted ChromaDB files...")
        try:
            shutil.rmtree(db_path)
            print("Successfully removed corrupted database")
        except Exception as e:
            print(f"Error removing database: {e}")
            return False
    
    print("Recreating ChromaDB with fresh indexes...")
    try:
        # Create new client
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Recreate collections based on backed up metadata
        for col_info in collections_info:
            try:
                collection = client.create_collection(
                    name=col_info['name'],
                    metadata=col_info.get('metadata', {})
                )
                print(f"Recreated collection '{col_info['name']}' (was {col_info['count']} documents)")
            except Exception as e:
                print(f"Error recreating collection {col_info['name']}: {e}")
        
        # If no collections info, create default ones
        if not collections_info:
            default_collections = ['books', 'essays', 'ncert', 'pyq']
            for name in default_collections:
                try:
                    collection = client.create_collection(
                        name=name,
                        metadata={"description": f"{name.title()} content"}
                    )
                    print(f"Created default collection '{name}'")
                except Exception as e:
                    print(f"Error creating default collection {name}: {e}")
        
        print("ChromaDB reset complete!")
        print("Note: You'll need to re-index your documents using rag_build.py")
        return True
        
    except Exception as e:
        print(f"Error recreating database: {e}")
        return False

def test_chromadb():
    """Test if ChromaDB is working properly"""
    try:
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        collections = client.list_collections()
        print(f"ChromaDB is working! Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col.name}: {col.count()} documents")
        
        return True
    except Exception as e:
        print(f"ChromaDB test failed: {e}")
        return False

if __name__ == "__main__":
    print("ChromaDB Corruption Fix Tool")
    print("=" * 40)
    
    # First test if DB is accessible
    if not test_chromadb():
        print("\nDatabase appears corrupted. Attempting to fix...")
        if reset_chromadb():
            print("\nTesting fixed database...")
            test_chromadb()
        else:
            print("Failed to fix database. Manual intervention may be required.")
    else:
        print("\nDatabase appears to be working fine!")