import os
import asyncio
import hashlib
import logging
from dotenv import load_dotenv
from utils import (
    extract_text_from_file,
    chunk_text,
    embed_and_store_chunks_with_checkpoints,
    test_pinecone_search,
    DATASET_DIR
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Chunking")

# Set your input document path(s) here:
FILES = ["combined.pdf"]   # Add any more files as needed

async def chunk_and_embed_files():
    for filename in FILES:
        path = os.path.join(DATASET_DIR, filename)
        logger.info(f"Processing file: {path}")
        text = extract_text_from_file(path)
        doc_hash = hashlib.md5(filename.encode()).hexdigest()
        chunks = chunk_text(text, chunk_size=512, overlap=100, base_hash=doc_hash)
        logger.info(f"Created {len(chunks)} chunks for {filename}.")
        
        # Use local embedding - much faster!
        checkpoint_file = f"embedding_progress_{doc_hash}.json"
        await embed_and_store_chunks_with_checkpoints(chunks, checkpoint_file)
        logger.info(f"🎉 Embedding & Upsert done for {filename}!")

if __name__ == "__main__":
    print("🚀 Starting local embedding process...")
    asyncio.run(chunk_and_embed_files())
    print("✅ All documents embedded!")
    
    print("\n🔍 Testing search functionality...")
    asyncio.run(test_pinecone_search())
    
    print("\n🎯 Ready to serve queries!")







