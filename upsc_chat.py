import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def basic_query_with_llm():
    """Query ChromaDB and generate answers using OpenAI"""
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize OpenAI (v1 client API)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No OpenAI API key found. Set OPENAI_API_KEY in .env file")
        return

    openai_client = OpenAI(api_key=api_key)
    
    # Get collections
    collections = {
        'books': client.get_collection(name="books"),
        'essays': client.get_collection(name="essays"),
        'ncert': client.get_collection(name="ncert"),
        'pyq': client.get_collection(name="pyq")
    }
    
    print("Available collections:")
    for name, collection in collections.items():
        print(f"  - {name}: {collection.count()} documents")
    
    # Interactive query loop
    while True:
        query = input("\nEnter your UPSC question (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            continue
        
        print(f"\nSearching for: '{query}'")
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query]).tolist()[0]
        
        # Collect relevant content
        all_content = []
        
        # Search in all collections
        for collection_name, collection in collections.items():
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,  # Get top 2 results per collection
                    include=['documents', 'metadatas', 'distances']
                )
                
                if results['documents'][0]:
                    for doc, metadata, distance in zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    ):
                        if distance < 0.7:  # Only include relevant results
                            source = metadata.get('source', '') if metadata else ''
                            all_content.append({
                                'content': doc,
                                'source': f"{collection_name.upper()}: {source}",
                                'relevance': 1 - distance
                            })
                            
            except Exception as e:
                print(f"Error searching {collection_name}: {e}")
        
        if not all_content:
            print("No relevant content found.")
            continue
        
        # Sort by relevance and take top 5
        all_content.sort(key=lambda x: x['relevance'], reverse=True)
        top_content = all_content[:5]
        
        # Prepare context for LLM
        context = "=== RELEVANT UPSC STUDY MATERIAL ===\n\n"
        for i, item in enumerate(top_content, 1):
            context += f"Source {i}: {item['source']}\n"
            context += f"Content: {item['content'][:800]}...\n\n"
        
        # Generate answer using OpenAI
        print("Generating comprehensive answer...")
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert UPSC preparation tutor. Use the provided study materials to give comprehensive, structured answers. 

Guidelines:
- Provide detailed explanations suitable for UPSC preparation
- Use bullet points and clear structure
- Include examples when possible
- Cite which sources you used
- If information is insufficient, mention what's missing
- Format answers clearly for easy reading"""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\n{context}\n\nPlease provide a comprehensive answer based on the above materials."
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )

            # OpenAI v1: message is an object with .content
            answer = response.choices[0].message.content
            
            print("\n" + "="*60)
            print("📚 COMPREHENSIVE ANSWER:")
            print("="*60)
            print(answer)
            print("="*60)
            
            print(f"\n🔍 SOURCES USED: {len(top_content)} documents")
            for item in top_content[:3]:  # Show top 3 sources
                print(f"  • {item['source']} (Relevance: {item['relevance']:.3f})")
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            print("Showing search results instead:")
            
            # Fallback to showing search results
            for i, item in enumerate(top_content[:3], 1):
                print(f"\nResult {i} (Relevance: {item['relevance']:.3f}):")
                print(f"Source: {item['source']}")
                print(f"Content: {item['content'][:300]}...")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    try:
        basic_query_with_llm()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set OPENAI_API_KEY in your .env file")
        print("2. Ensure ChromaDB is set up properly")