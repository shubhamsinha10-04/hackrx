import os
import io
import re
import hashlib
import json
import asyncio
from typing import List, Dict
from pydantic import BaseModel
import logging
import PyPDF2
from docx import Document
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import httpx
from sentence_transformers import SentenceTransformer

load_dotenv()

# ENV
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
DIMENSION = 384  # Changed from 3072 to 384 for sentence-transformers

# Initialize sentence transformer model (downloads ~90MB first time)
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded successfully!")

# Pinecone setup - will recreate index with new dimension
pc = Pinecone(api_key=PINECONE_API_KEY)

# Delete existing index if it has wrong dimension
if INDEX_NAME in pc.list_indexes().names():
    existing_index = pc.describe_index(INDEX_NAME)
    if existing_index.dimension != DIMENSION:
        print(f"Deleting existing index (dimension {existing_index.dimension} → {DIMENSION})")
        pc.delete_index(INDEX_NAME)

# Create index with correct dimension
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index with dimension {DIMENSION}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

Index = pc.Index(INDEX_NAME)

logger = logging.getLogger("hackrx_utils")
logging.basicConfig(level=logging.INFO)

class TextChunk(BaseModel):
    id: str
    text: str

#################
# CHUNKING utils
#################
def extract_text_from_file(filepath: str) -> str:
    """Extracts all text from a PDF or DOCX file on disk."""
    if filepath.lower().endswith(".pdf"):
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
    elif filepath.lower().endswith(".docx"):
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    raise Exception("Unsupported file type (only PDF and DOCX allowed)")

def chunk_text(text: str, chunk_size=512, overlap=100, base_hash=None) -> List[TextChunk]:
    text = re.sub(r"\\s+", " ", text.strip())
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        unique_id = f"{base_hash}_chunk_{i}"
        chunks.append(TextChunk(id=unique_id, text=" ".join(chunk_words)))
        i += chunk_size - overlap
    return chunks

#########################
# LOCAL EMBEDDING utils (NO API calls, NO rate limits!)
#########################
async def get_local_embedding(text: str) -> List[float]:
    """Get embedding using local sentence-transformers - completely free and fast!"""
    loop = asyncio.get_event_loop()
    # Run model inference in thread pool to avoid blocking
    embedding = await loop.run_in_executor(None, embedding_model.encode, text)
    return embedding.tolist()

async def embed_and_store_chunks_with_checkpoints(chunks: List[TextChunk], checkpoint_file="embedding_progress.json", batch_size: int = 50):
    """Embed chunks locally - no rate limits, much faster!"""
    # Load checkpoint if exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('last_processed', 0)
        logger.info(f"Resuming from chunk {start_idx}")
    
    vectors = []
    for i, chunk in enumerate(chunks[start_idx:], start_idx):
        try:
            # Local embedding - instant, no rate limits!
            emb = await get_local_embedding(chunk.text)
            vectors.append({"id": chunk.id, "values": emb, "metadata": {"text": chunk.text}})
            logger.info(f"✅ Embedded chunk {chunk.id} ({i+1}/{len(chunks)})")
            
            # Save checkpoint every 25 chunks (faster than API)
            if (i + 1) % 25 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_processed': i + 1}, f)
                logger.info(f"Checkpoint saved at chunk {i+1}")
            
        except Exception as e:
            logger.error(f"Failed to embed chunk {chunk.id}: {e}")
            continue
            
        # No rate limiting needed - local processing is instant!
        await asyncio.sleep(0.01)
        
        if len(vectors) >= batch_size:
            Index.upsert(vectors=vectors)
            logger.info(f"📤 Upserted {len(vectors)} vectors to Pinecone")
            vectors = []
    
    if vectors:
        Index.upsert(vectors=vectors)
        logger.info(f"📤 Upserted final {len(vectors)} vectors to Pinecone")
    
    # Clean up checkpoint file when done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("🎉 Embedding complete! Checkpoint file removed.")

# Legacy function for backward compatibility
async def embed_and_store_chunks(chunks: List[TextChunk], batch_size: int = 50):
    """Legacy function - calls the checkpoint version."""
    await embed_and_store_chunks_with_checkpoints(chunks, "embedding_progress_legacy.json", batch_size)

####################################
# QUERY/ANSWERING (used in FastAPI) with DEBUGGING
####################################
async def answer_questions(questions: List[str]) -> List[str]:
    answers = []
    for q in questions:
        contexts = await retrieve_context(q)
        answers.append(await generate_answer_openrouter(q, contexts))
    return answers

async def retrieve_context(query: str, top_k=10) -> List[str]:
    """Retrieve context with debugging - reduced top_k for better relevance"""
    emb = await get_local_embedding(query)
    res = Index.query(
        vector=emb,
        top_k=top_k,
        include_metadata=True
    )
    
    # DEBUG: Print search results
    print(f"\n🔍 Query: {query}")
    print(f"📊 Found {len(res['matches'])} matches")
    
    if res['matches']:
        # Show top 3 matches with similarity scores
        for i, match in enumerate(res['matches'][:3]):
            score = match.get('score', 0)
            text_preview = match['metadata']['text'][:100].replace('\n', ' ') + "..."
            print(f"  Match {i+1}: Score={score:.3f} | {text_preview}")
    else:
        print("❌ No matches found!")
        return []
    
    # Filter for better similarity (optional)
    relevant_matches = [m for m in res['matches'] if m.get('score', 0) > 0.3]
    if not relevant_matches:
        print("⚠️ No matches above 0.3 similarity, using top results anyway")
        relevant_matches = res['matches'][:5]
    
    contexts = [match['metadata']['text'] for match in relevant_matches]
    total_chars = len(' '.join(contexts))
    print(f"📝 Using {len(contexts)} chunks, {total_chars} characters total")
    
    return contexts

async def generate_answer_openrouter(question: str, contexts: List[str]) -> str:
    """Generate answer with debugging"""
    if not contexts:
        print("❌ No context provided to LLM")
        return "No relevant information found in the document."
    
    context_text = "\n\n".join(contexts)
    print(f"\n💭 Generating answer for: {question}")
    print(f"📄 Context chunks: {len(contexts)}")
    print(f"📏 Context length: {len(context_text)} chars")
    
    # Show first 200 chars of context for debugging
    preview = context_text[:200].replace('\n', ' ') + "..." if len(context_text) > 200 else context_text
    print(f"🔤 Context preview: {preview}")
    
    prompt = f"""You are an expert insurance policy assistant. Based solely on the following context, answer briefly and precisely:

CONTEXT:
{context_text}

QUESTION:
{question}

If answer is unavailable, respond "Information not available in provided context."."""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You answer insurance policy queries factually and concisely."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 250,
        "temperature": 0.1
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            logger.error(f"OpenRouter error {resp.status_code}: {resp.text}")
            return "Error: Unable to get answer from LLM."
        data = resp.json()
    
    try:
        answer = data["choices"][0]["message"]["content"].strip()
        print(f"🤖 LLM Response: {answer}")
        return answer
    except Exception:
        return "Error: Malformed LLM response."

####################################
# TEST FUNCTION to verify Pinecone data
####################################
async def test_pinecone_search():
    """Test function to verify chunks are properly stored and searchable"""
    # Get index stats
    try:
        stats = Index.describe_index_stats()
        print(f"\n📊 Index Stats:")
        print(f"  Total vectors: {stats.total_vector_count}")
        print(f"  Dimension: {stats.dimension}")
        if hasattr(stats, 'namespaces'):
            print(f"  Namespaces: {stats.namespaces}")
    except Exception as e:
        print(f"❌ Failed to get index stats: {e}")
        return
    
    # Test search with simple terms
    test_queries = ["insurance", "policy", "coverage", "premium"]
    
    for test_query in test_queries:
        try:
            emb = await get_local_embedding(test_query)
            results = Index.query(vector=emb, top_k=3, include_metadata=True)
            
            print(f"\n🔍 Test search for '{test_query}':")
            if results['matches']:
                for i, match in enumerate(results['matches']):
                    score = match.get('score', 0)
                    text = match['metadata']['text'][:80].replace('\n', ' ')
                    print(f"  {i+1}. Score: {score:.3f} | {text}...")
            else:
                print(f"  ❌ No results for '{test_query}'")
        except Exception as e:
            print(f"❌ Search failed for '{test_query}': {e}")








