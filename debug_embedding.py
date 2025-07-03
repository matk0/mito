#!/usr/bin/env python3

import json
import time
import sys
from pathlib import Path

def test_step(step_name, func):
    """Test a step and report timing/success."""
    print(f"\n🔍 Testing: {step_name}")
    print("-" * 50)
    start_time = time.time()
    try:
        result = func()
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {step_name} completed in {elapsed:.1f}s")
        return result, True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {step_name} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return None, False

def test_imports():
    """Test all required imports."""
    print("Importing required modules...")
    import numpy as np
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    print("All imports successful")
    return True

def test_file_loading():
    """Test loading the chunked data file."""
    input_file = "./chunked_data/chunked_content.json"
    print(f"Loading file: {input_file}")
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def test_model_loading():
    """Test loading the embedding model."""
    from sentence_transformers import SentenceTransformer
    
    model_name = "intfloat/multilingual-e5-large"
    print(f"Loading model: {model_name}")
    print("This may take time if downloading...")
    
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model

def test_single_embedding(model):
    """Test generating a single embedding."""
    test_text = "passage: Toto je test text pre embedding."
    print(f"Generating embedding for: '{test_text[:50]}...'")
    
    embedding = model.encode([test_text])
    print(f"Embedding shape: {embedding.shape}")
    return embedding

def test_batch_embedding(model, chunks):
    """Test generating embeddings for a small batch."""
    batch_size = 5
    test_chunks = chunks[:batch_size]
    texts = [f"passage: {chunk['text'][:100]}" for chunk in test_chunks]
    
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings")
    return embeddings

def test_chromadb_init():
    """Test ChromaDB initialization."""
    import chromadb
    from chromadb.config import Settings
    
    db_path = "./test_vector_db"
    Path(db_path).mkdir(exist_ok=True)
    
    print(f"Initializing ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = "test_collection"
    try:
        collection = client.get_collection(collection_name)
        print(f"Found existing collection: {collection_name}")
    except:
        collection = client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")
    
    return client, collection

def test_chromadb_storage(collection, embeddings):
    """Test storing embeddings in ChromaDB."""
    print("Testing ChromaDB storage...")
    
    # Test data
    ids = [f"test_chunk_{i}" for i in range(len(embeddings))]
    documents = [f"Test document {i}" for i in range(len(embeddings))]
    metadatas = [{"test": True, "id": i} for i in range(len(embeddings))]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    
    count = collection.count()
    print(f"Stored and verified {count} items in ChromaDB")
    return True

def main():
    print("🔧 EMBEDDING SCRIPT DIAGNOSTIC")
    print("=" * 60)
    
    # Test 1: Imports
    _, success = test_step("Import modules", test_imports)
    if not success:
        return
    
    # Test 2: File loading
    chunks, success = test_step("Load chunked data", test_file_loading)
    if not success:
        return
    
    # Test 3: Model loading (this is likely where it hangs)
    model, success = test_step("Load embedding model", test_model_loading)
    if not success:
        return
    
    # Test 4: Single embedding
    embedding, success = test_step("Generate single embedding", lambda: test_single_embedding(model))
    if not success:
        return
    
    # Test 5: Batch embedding
    embeddings, success = test_step("Generate batch embeddings", lambda: test_batch_embedding(model, chunks))
    if not success:
        return
    
    # Test 6: ChromaDB initialization
    client_collection, success = test_step("Initialize ChromaDB", test_chromadb_init)
    if not success:
        return
    client, collection = client_collection
    
    # Test 7: ChromaDB storage
    _, success = test_step("Store in ChromaDB", lambda: test_chromadb_storage(collection, embeddings))
    if not success:
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("The embedding script should work properly.")
    print("=" * 60)

if __name__ == "__main__":
    main()