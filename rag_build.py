import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict, Any, Optional
import hashlib
import re
from pathlib import Path


# UPSC subject/topic taxonomy derived from your syllabus structure
UPSC_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "HISTORY": {
        "Ancient India": [
            "Prehistoric", "Protohistoric", "Paleolithic", "Mesolithic", "Neolithic",
            "Chalcolithic cultures", "Indus Valley Civilization", "Harappan",
            "Town planning", "Harappan town planning", "Harappan economy",
            "Harappan religion", "Harappan script", "Decline theories", "Vedic Age",
            "Early Vedic", "Later Vedic", "Vedic society", "Sabha", "Samiti",
            "Vedic economy", "Vedic religion", "Vedic social system", "Mahajanapadas",
            "Magadha", "Jainism", "Buddhism", "Four Noble Truths", "Religious movements",
            "Mauryan Empire", "Ashoka", "Dhamma", "Mauryan administration",
            "Gupta Age", "Gupta administration", "Gupta economy",
            "Gupta science", "Gupta mathematics", "Gupta art", "Gupta architecture",
        ],
        "Medieval India": [
            "Delhi Sultanate", "Slave Dynasty", "Khalji", "Khilji", "Tughlaq",
            "Iqta system", "Mughal Empire", "Mansabdari", "Jagirdari",
            "Mughal religious policy", "Mughal art", "Mughal architecture",
            "Bhakti movement", "Sufi movement", "Bhakti saints", "Sufi saints",
        ],
        "Modern India": [
            "Advent of Europeans", "Portuguese", "Dutch", "French", "British",
            "Subsidiary Alliance", "Doctrine of Lapse", "Land revenue systems",
            "Revolt of 1857", "1857 revolt", "Nature of 1857",
            "Consequences of 1857", "National Movement", "Moderates", "Extremists",
            "Gandhian movements", "Non cooperation", "Civil Disobedience", "Quit India",
            "Revolutionary movements", "Constitutional reforms",
        ],
    },
    "POLITY": {
        "Constitution": [
            "Constitution of India", "Features of the Constitution", "Constitutional amendments",
            "Schedules of the Constitution", "Sources of the Constitution",
        ],
        "Fundamental Rights": [
            "Article 12", "Article 13", "Article 14", "Article 15", "Article 16",
            "Article 19", "Article 21", "Article 32", "Fundamental Rights", "Writs",
            "Habeas Corpus", "Mandamus", "Certiorari", "Prohibition", "Quo Warranto",
        ],
        "Parliament": [
            "Parliament", "Lok Sabha", "Rajya Sabha", "Legislative procedure",
            "Money Bill", "Parliamentary committees", "Standing Committee",
        ],
        "Executive": [
            "President of India", "Prime Minister", "Council of Ministers", "Cabinet",
        ],
        "Judiciary": [
            "Judicial review", "Public Interest Litigation", "PIL", "Collegium",
            "Supreme Court", "High Court",
        ],
        "Federalism": [
            "Centre-State relations", "Inter state relations", "Emergency provisions",
            "President's Rule", "Financial emergency",
        ],
        "Constitutional Bodies": [
            "Election Commission", "CAG", "Comptroller and Auditor General", "UPSC",
        ],
    },
    "GEOGRAPHY": {
        "Physical Geography": [
            "Plate tectonics", "Volcanoes", "Earthquakes", "Cyclones", "Ocean currents",
        ],
        "Indian Geography": [
            "Indian monsoon", "Monsoon mechanism", "Himalayan rivers", "Peninsular rivers",
            "River systems", "Indian soils", "Black soil", "Alluvial soil", "Agriculture patterns",
        ],
    },
    "ECONOMY": {
        "Basic Concepts": [
            "GDP", "Gross Domestic Product", "Inflation", "Fiscal deficit",
            "Monetary policy", "Fiscal policy",
        ],
        "Banking": [
            "Reserve Bank of India", "RBI", "Repo rate", "Reverse repo", "NPA",
            "Non performing asset", "Banking sector",
        ],
        "Government Schemes": [
            "Inclusive growth", "MSME", "Micro Small and Medium Enterprises",
            "Agriculture subsidies", "Government schemes",
        ],
    },
    "ENVIRONMENT": {
        "Environment & Ecology": [
            "Ecosystem structure", "Food chain", "Food web", "Ecological pyramid",
            "Biodiversity hotspots", "Climate change", "UNFCCC", "Paris Agreement",
            "National parks", "Wildlife sanctuary", "Biosphere reserve",
        ],
    },
    "SCIENCE & TECH": {
        "Science & Technology": [
            "Biotechnology", "Genetic engineering", "Space technology", "ISRO",
            "AI", "Artificial Intelligence", "Machine Learning", "Cybersecurity",
            "Quantum computing",
        ],
    },
    "INTERNATIONAL RELATIONS": {
        "International Relations": [
            "India USA", "India US", "India America", "India China", "India Russia",
            "QUAD", "BRICS", "UN reforms", "United Nations reforms",
        ],
    },
    "ETHICS": {
        "Ethics": [
            "Ethics definitions", "Attitude", "Emotional intelligence",
            "Civil service values", "Case studies framework",
        ],
    },
}

class UPSCRAGSystem:
    def __init__(self, openai_api_key: str = None):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client (supports openai>=1.0 style)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm_client: Optional[OpenAI] = OpenAI(api_key=api_key)
            self.use_openai = True
        else:
            self.llm_client = None
            self.use_openai = False
        
        # Create collections for different content types
        self.collections = {
            'books': self.client.get_or_create_collection(
                name="books",
                metadata={"description": "Book chapters content"}
            ),
            'essays': self.client.get_or_create_collection(
                name="essays", 
                metadata={"description": "Essay titles and content"}
            ),
            'ncert': self.client.get_or_create_collection(
                name="ncert",
                metadata={"description": "NCERT topics and content"}
            ),
            'pyq': self.client.get_or_create_collection(
                name="pyq",
                metadata={"description": "Previous year questions and answers"}
            )
        }

    def _classify_upsc_tags(self, text: str, hints: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
        """Classify a text chunk into UPSC subject/topic/subtopic using simple keyword heuristics.

        We combine the chunk text with optional high-level hints (book name, chapter,
        given subject/category, PYQ topic, etc.) and then look for the most specific
        matching phrase from the UPSC_TAXONOMY.
        """

        combined = text or ""
        if hints:
            combined = " \n".join([h for h in hints if h]) + " \n" + combined

        text_lower = combined.lower()
        best_subject: Optional[str] = None
        best_topic: Optional[str] = None
        best_subtopic: Optional[str] = None
        best_score = 0

        for subject, topics in UPSC_TAXONOMY.items():
            for topic, subtopics in topics.items():
                for sub in subtopics:
                    key = sub.lower()
                    if not key:
                        continue
                    if key in text_lower:
                        # Prefer longer/more specific matches
                        score = len(key)
                        if score > best_score:
                            best_score = score
                            best_subject = subject
                            best_topic = topic
                            best_subtopic = sub

        return {
            "upsc_subject": best_subject,
            "upsc_topic": best_topic,
            "upsc_subtopic": best_subtopic,
        }

    def chunk_text(self, text: str, max_chunk_size: int = 900, overlap: int = 180) -> List[str]:
        """Paragraph/line-aware chunking with light overlap.

        Strategy:
        - Prefer paragraph boundaries (split on blank lines).
        - For very long paragraphs, further split by sentences.
        - For bullet-style content, respect line boundaries but merge short lines.
        - Then build chunks up to max_chunk_size characters, with overlap from
          the previous chunk (last sentences or tail text).
        """

        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= max_chunk_size:
            return [text]

        # First split into paragraphs using blank lines
        raw_paragraphs = re.split(r"\n\s*\n+", text)
        paragraphs: List[str] = [p.strip() for p in raw_paragraphs if p.strip()]

        segments: List[str] = []
        for para in paragraphs:
            # If paragraph is extremely long, split further by sentences
            if len(para) > max_chunk_size * 1.5:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
                segments.extend(sentences)
            else:
                # For note-style paragraphs with many short lines, split by lines
                if "\n" in para and len(para) <= max_chunk_size:
                    lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
                    current = ""
                    for ln in lines:
                        if len(current) + len(ln) + 1 <= max_chunk_size:
                            current = (current + " " + ln).strip()
                        else:
                            if current:
                                segments.append(current)
                            current = ln
                    if current:
                        segments.append(current)
                else:
                    segments.append(para)

        # Now accumulate segments into chunks with a soft size limit
        chunks: List[str] = []
        current_chunk = ""
        for seg in segments:
            if not current_chunk:
                current_chunk = seg
            elif len(current_chunk) + 2 + len(seg) <= max_chunk_size:
                current_chunk = current_chunk + "\n\n" + seg
            else:
                chunks.append(current_chunk.strip())
                current_chunk = seg
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap using tail of previous chunk (by sentences where possible)
        overlapped_chunks: List[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                prev = chunks[i - 1]
                # Try to take last 1–2 sentences from previous chunk
                prev_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prev) if s.strip()]
                if len(prev_sentences) >= 2:
                    tail = " ".join(prev_sentences[-2:])
                elif prev_sentences:
                    tail = prev_sentences[-1]
                else:
                    tail = prev[-overlap:] if len(prev) > overlap else prev
                chunk = tail + "\n\n" + chunk
            overlapped_chunks.append(chunk)

        return overlapped_chunks

    def load_books_data(self, books_folder: str):
        """Load and index books data"""
        books_path = Path(books_folder)
        
        for book_file in books_path.glob("*.json"):
            print(f"Processing {book_file.name}...")
            
            with open(book_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            book_name = book_file.stem  # filename without extension
            
            for idx, item in enumerate(data):
                chapter = item.get('chapter', f'Chapter {idx+1}')
                content = item.get('content', '')
                
                # Chunk the content (book-style text, slightly smaller chunks)
                chunks = self.chunk_text(content, max_chunk_size=900, overlap=180)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # Create unique ID
                    chunk_id = hashlib.md5(f"{book_name}_{chapter}_{chunk_idx}".encode()).hexdigest()

                    # UPSC subject/topic classification
                    upsc_tags = self._classify_upsc_tags(chunk, hints=[book_name, chapter])

                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk).tolist()

                    # Add to collection
                    metadata = {
                        'source': 'books',
                        'book': book_name,
                        'chapter': chapter,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                    }
                    metadata.update(upsc_tags)
                    # Chroma metadata cannot contain None values
                    clean_metadata = {k: v for k, v in metadata.items() if v is not None}

                    self.collections['books'].add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[clean_metadata],
                        ids=[chunk_id]
                    )

    def load_essays_data(self, essays_file: str):
        """Load and index essays data"""
        print(f"Processing essays...")
        
        with open(essays_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            title = item.get('title', f'Essay {idx+1}')
            content = item.get('content', '')
            
            # Chunk the content (essays can be longer, but still paragraph-aware)
            chunks = self.chunk_text(content, max_chunk_size=1200, overlap=200)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = hashlib.md5(f"essay_{title}_{chunk_idx}".encode()).hexdigest()

                # UPSC subject/topic classification (title is a strong hint)
                upsc_tags = self._classify_upsc_tags(chunk, hints=[title])

                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()

                # Add to collection
                metadata = {
                    'source': 'essays',
                    'title': title,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                }
                metadata.update(upsc_tags)
                clean_metadata = {k: v for k, v in metadata.items() if v is not None}

                self.collections['essays'].add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[clean_metadata],
                    ids=[chunk_id]
                )

    def load_ncert_data(self, ncert_file: str):
        """Load and index NCERT data"""
        print(f"Processing NCERT topics...")
        
        with open(ncert_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            subject = item.get('subject', 'Unknown')
            category = item.get('category', 'Unknown')
            content = item.get('content', '')
            
            # Chunk the content (NCERT-style text)
            chunks = self.chunk_text(content, max_chunk_size=900, overlap=180)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = hashlib.md5(f"ncert_{subject}_{category}_{chunk_idx}".encode()).hexdigest()

                # UPSC subject/topic classification (subject/category are strong hints)
                upsc_tags = self._classify_upsc_tags(chunk, hints=[subject, category])

                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()

                # Add to collection
                metadata = {
                    'source': 'ncert',
                    'subject': subject,
                    'category': category,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                }
                metadata.update(upsc_tags)
                clean_metadata = {k: v for k, v in metadata.items() if v is not None}

                self.collections['ncert'].add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[clean_metadata],
                    ids=[chunk_id]
                )

    def load_pyq_data(self, pyq_file: str):
        """Load and index PYQ data"""
        print(f"Processing PYQ...")
        
        with open(pyq_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data):
            question = item.get('question', '')
            answer = item.get('answer', '')
            year = item.get('year', 'Unknown')
            topic = item.get('topic', 'Unknown')
            
            # Combine question and answer for better context
            full_content = f"Question: {question}\n\nAnswer: {answer}"
            
            # Chunk the content (PYQ answers can be very long)
            chunks = self.chunk_text(full_content, max_chunk_size=1000, overlap=200)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID
                chunk_id = hashlib.md5(f"pyq_{year}_{topic}_{idx}_{chunk_idx}".encode()).hexdigest()

                # UPSC subject/topic classification (PYQ topic and question are strong hints)
                upsc_tags = self._classify_upsc_tags(chunk, hints=[topic, question])

                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()

                # Add to collection
                metadata = {
                    'source': 'pyq',
                    'question': question[:200] + "..." if len(question) > 200 else question,  # Truncate for metadata
                    'year': year,
                    'topic': topic,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                }
                metadata.update(upsc_tags)
                clean_metadata = {k: v for k, v in metadata.items() if v is not None}

                self.collections['pyq'].add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[clean_metadata],
                    ids=[chunk_id]
                )

    def search(self, query: str, n_results: int = 5, source_filter: List[str] = None) -> Dict[str, Any]:
        """Search across all collections"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        # Define collections to search
        collections_to_search = ['books', 'essays', 'ncert', 'pyq']
        if source_filter:
            collections_to_search = [c for c in collections_to_search if c in source_filter]
        
        # Search each collection
        for collection_name in collections_to_search:
            try:
                results = self.collections[collection_name].query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, 10)  # Limit per collection
                )
                
                # Process results
                for i in range(len(results['documents'][0])):
                    all_results.append({
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0,
                        'collection': collection_name
                    })
            except Exception as e:
                print(f"Error searching in {collection_name}: {e}")
        
        # Sort by similarity (lower distance = higher similarity)
        all_results.sort(key=lambda x: x['distance'])
        
        return {
            'query': query,
            'results': all_results[:n_results],
            'total_found': len(all_results)
        }

    def generate_answer(self, query: str, search_results: Dict[str, Any], model: str = "gpt-3.5-turbo") -> str:
        """Generate answer using LLM with retrieved context"""
        if not self.use_openai or self.llm_client is None:
            return "OpenAI API key not provided. Cannot generate LLM response."
        
        # Prepare context from search results
        context_parts = []
        for result in search_results['results'][:3]:  # Use top 3 results
            source_info = f"Source: {result['metadata']['source']}"
            if result['metadata']['source'] == 'books':
                source_info += f" - {result['metadata']['book']} ({result['metadata']['chapter']})"
            elif result['metadata']['source'] == 'essays':
                source_info += f" - {result['metadata']['title']}"
            elif result['metadata']['source'] == 'ncert':
                source_info += f" - {result['metadata']['subject']} ({result['metadata']['category']})"
            elif result['metadata']['source'] == 'pyq':
                source_info += f" - {result['metadata']['year']} ({result['metadata']['topic']})"
            
            context_parts.append(f"{source_info}\n{result['document']}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context from UPSC preparation materials, please provide a comprehensive answer to the question.

Context:
{context}

Question: {query}

Please provide a detailed answer that:
1. Directly addresses the question
2. Uses information from the provided context
3. Is suitable for UPSC exam preparation
4. Includes relevant examples and key points

Answer:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful UPSC exam preparation assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def ask_question(self, question: str, source_filter: List[str] = None, n_results: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline: search + generate"""
        # Search for relevant documents
        search_results = self.search(question, n_results, source_filter)
        
        # Generate answer using LLM
        answer = self.generate_answer(question, search_results)
        
        return {
            'question': question,
            'answer': answer,
            'sources': search_results['results'],
            'total_sources_found': search_results['total_found']
        }

# Usage example
def main():
    # Initialize RAG system
    rag = UPSCRAGSystem(openai_api_key="your-openai-api-key-here")  # Replace with your key
    
    # Load data
    rag.load_books_data("upsc_data/books")
    rag.load_essays_data("upsc_data/essay/essay_titles.json")
    rag.load_ncert_data("upsc_data/ncert/ncert_topics.json")
    rag.load_pyq_data("upsc_data/pyq/combined_pyq.json")
    
    # Test queries
    test_questions = [
        "What is the contribution of Pallavas to South Indian art?",
        "Explain the structure of public services in India",
        "What are the features of Indian federalism?",
        "Discuss the role of Lokpal in Indian governance"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
        
        result = rag.ask_question(question)
        print(f"Answer: {result['answer']}")
        
        print(f"\nSources used ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"{i}. {source['metadata']['source']}: {source['document'][:100]}...")

if __name__ == "__main__":
    main()