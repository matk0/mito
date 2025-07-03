#!/usr/bin/env python3

import json
from sentence_transformers import SentenceTransformer
import time

print("🤖 Testing embedding model loading...")
start_time = time.time()

# Try to load the model
model = SentenceTransformer("intfloat/multilingual-e5-large")
load_time = time.time() - start_time

print(f"✅ Model loaded in {load_time:.1f}s")
print(f"📏 Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Test a simple embedding
test_text = "Toto je test text pre embedding."
print(f"🔍 Testing embedding generation with: '{test_text}'")

embedding_start = time.time()
embedding = model.encode([test_text])
embedding_time = time.time() - embedding_start

print(f"✅ Embedding generated in {embedding_time:.3f}s")
print(f"📊 Embedding shape: {embedding.shape}")
print(f"🎯 First 5 values: {embedding[0][:5]}")

print("🎉 Test completed successfully!")