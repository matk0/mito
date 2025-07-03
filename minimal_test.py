#!/usr/bin/env python3

import time

print("Step 1: Basic imports...")
import json
import sys
from pathlib import Path
print("✅ Basic imports OK")

print("Step 2: NumPy import...")
import numpy as np
print("✅ NumPy OK")

print("Step 3: ChromaDB import...")
try:
    import chromadb
    print("✅ ChromaDB OK")
except Exception as e:
    print(f"❌ ChromaDB failed: {e}")

print("Step 4: SentenceTransformers import...")
try:
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers import OK")
except Exception as e:
    print(f"❌ SentenceTransformers failed: {e}")

print("Step 5: Model loading test...")
try:
    print("Loading model (this may hang)...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

print("Test complete!")