#!/usr/bin/env python3

import json
import os
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
import logging
from datetime import datetime

class EmbeddingGenerator:
    def __init__(self, 
                 input_file: str = "./chunked_data/chunked_content.json",
                 db_path: str = "./vector_db",
                 model_name: str = "intfloat/multilingual-e5-large",
                 collection_name: str = "slovak_blog_chunks"):
        """
        Initialize the embedding generator and vector database.
        
        Args:
            input_file: Path to chunked content JSON file
            db_path: Directory for Chroma vector database
            model_name: SentenceTransformer model for embeddings
            collection_name: Name for the Chroma collection
        """
        self.input_file = Path(input_file)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.collection_name = collection_name
        
        # Create vector database directory
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        print(f"🤖 Loading embedding model: {model_name}")
        print(f"⏳ This may take a few minutes if downloading for the first time (~1.1GB)...")
        model_start = time.time()
        self.model = SentenceTransformer(model_name)
        model_load_time = time.time() - model_start
        print(f"✅ Model loaded in {model_load_time:.1f}s. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Initialize Chroma client
        print(f"🗄️ Initializing vector database at: {self.db_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"📁 Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Slovak blog content chunks with multilingual embeddings"}
            )
            print(f"📁 Created new collection: {collection_name}")
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunked content from JSON file."""
        print(f"📂 Loading chunks from: {self.input_file}")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            metadata = data.get('metadata', {})
            
            print(f"✅ Loaded {len(chunks)} chunks")
            print(f"📊 Chunk statistics: {metadata.get('statistics', {})}")
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return []
    
    def prepare_text_for_embedding(self, text: str) -> str:
        """
        Prepare text for embedding generation.
        Add query prefix for multilingual-e5 models to improve performance.
        """
        # For multilingual-e5 models, prefixing with "query: " or "passage: " can improve performance
        # We use "passage: " since these are document chunks
        return f"passage: {text.strip()}"
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"🔄 Generating embeddings for {len(texts)} texts in {total_batches} batches of {batch_size}...")
        embedding_start = time.time()
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"⚡ Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            batch_start = time.time()
            
            prepared_batch = [self.prepare_text_for_embedding(text) for text in batch]
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                prepared_batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            
            batch_time = time.time() - batch_start
            print(f"✅ Batch {batch_num} completed in {batch_time:.1f}s")
            
            embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        all_embeddings = np.vstack(embeddings)
        total_embedding_time = time.time() - embedding_start
        print(f"✅ Generated {len(all_embeddings)} embeddings in {total_embedding_time:.1f}s")
        
        return all_embeddings
    
    def store_in_vector_db(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Store chunks and embeddings in Chroma vector database."""
        print(f"💾 Storing {len(chunks)} embeddings in vector database...")
        storage_start = time.time()
        
        # Prepare data for Chroma
        ids = []
        documents = []
        metadatas = []
        embeddings_list = embeddings.tolist()
        
        for i, chunk in enumerate(chunks):
            # Create unique ID
            chunk_id = f"chunk_{chunk.get('global_chunk_id', i)}"
            ids.append(chunk_id)
            
            # Document text
            documents.append(chunk['text'])
            
            # Metadata (Chroma doesn't support nested objects, so flatten)
            metadata = {
                'source_url': chunk.get('source_url', ''),
                'source_title': chunk.get('source_title', ''),
                'source_date': chunk.get('source_date', ''),
                'word_count': chunk.get('word_count', 0),
                'chunk_id': chunk.get('chunk_id', 0),
                'global_chunk_id': chunk.get('global_chunk_id', i),
                'article_excerpt': chunk.get('article_excerpt', '')[:200]  # Truncate long excerpts
            }
            metadatas.append(metadata)
        
        # Store in collection (batch upsert)
        batch_size = 100  # Chroma batch size limit
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        print(f"📦 Storing in {total_batches} batches of {batch_size} items each...")
        
        for i in tqdm(range(0, len(ids), batch_size), desc="Storing in database"):
            batch_end = min(i + batch_size, len(ids))
            batch_num = (i // batch_size) + 1
            
            print(f"💽 Storing batch {batch_num}/{total_batches} ({batch_end - i} items)...")
            
            self.collection.upsert(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                embeddings=embeddings_list[i:batch_end]
            )
        
        storage_time = time.time() - storage_start
        print(f"✅ Stored {len(ids)} chunks in vector database in {storage_time:.1f}s")
    
    def test_similarity_search(self, query: str = "mitochondria a energia", top_k: int = 3):
        """Test the vector database with a similarity search."""
        print(f"\n🔍 Testing similarity search with query: '{query}'")
        
        # Prepare query for embedding
        prepared_query = self.prepare_text_for_embedding(query)
        
        # Generate query embedding
        query_embedding = self.model.encode([prepared_query], normalize_embeddings=True)[0]
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"📊 Found {len(results['documents'][0])} results:")
        print("-" * 80)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"\n🔹 Result #{i+1} (Similarity: {similarity:.3f})")
            print(f"📄 Title: {metadata['source_title']}")
            print(f"📅 Date: {metadata['source_date']}")
            print(f"🔗 URL: {metadata['source_url']}")
            print(f"📝 Preview: {doc[:200]}...")
            print("-" * 40)
    
    def get_database_stats(self):
        """Get statistics about the vector database."""
        count = self.collection.count()
        print(f"\n📊 Vector Database Statistics:")
        print(f"📁 Collection: {self.collection_name}")
        print(f"📄 Total chunks stored: {count}")
        print(f"🤖 Embedding model: {self.model_name}")
        print(f"📏 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        print(f"💾 Database path: {self.db_path}")
    
    def process_all(self):
        """Main method to process all chunks and create embeddings."""
        start_time = time.time()
        
        print("🚀 Starting embedding generation process...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Load chunks
        chunks = self.load_chunks()
        if not chunks:
            print("❌ No chunks to process!")
            return
        
        # Extract text for embedding
        texts = [chunk['text'] for chunk in chunks]
        print(f"📝 Extracted text from {len(texts)} chunks")
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size=16)
        
        # Store in vector database
        self.store_in_vector_db(chunks, embeddings)
        
        # Test search functionality
        self.test_similarity_search("kvantová biológia a mitochondrie")
        self.test_similarity_search("cirkadiálny rytmus a svetlo")
        
        # Show statistics
        self.get_database_stats()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"🏁 Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print("✅ EMBEDDING GENERATION COMPLETE!")


def main():
    """Main function to run the embedding generation process."""
    print("Slovak Blog Embedding Generator")

    print("=" * 40)
    
    # Check if required files exist
    input_file = "./chunked_data/chunked_content.json"
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please run content_chunker.py first!")
        return
    
    # Initialize and run embedding generator
    generator = EmbeddingGenerator(
        input_file=input_file,
        db_path="./vector_db",
        model_name="intfloat/multilingual-e5-large",
        collection_name="slovak_blog_chunks"
    )
    
    # Process all chunks
    generator.process_all()


if __name__ == "__main__":
    main()
