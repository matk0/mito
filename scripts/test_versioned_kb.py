#!/usr/bin/env python3

"""
Test script for the versioned knowledge graph builder.
This script tests creating a KB from cleaned blog articles.
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🧪 Testing Versioned Knowledge Graph Builder")
    print("=" * 50)
    
    # Check if cleaned articles exist
    cleaned_articles_dir = Path("data/raw/scraped_data/articles/cleaned_articles/articles/")
    if not cleaned_articles_dir.exists():
        print(f"❌ Cleaned articles directory not found: {cleaned_articles_dir}")
        print("Please ensure the articles have been cleaned first.")
        return False
    
    # Count files
    json_files = list(cleaned_articles_dir.glob("*.json"))
    print(f"📂 Found {len(json_files)} cleaned article files")
    
    if len(json_files) == 0:
        print("❌ No JSON files found in cleaned articles directory")
        return False
    
    # Test the versioned KB builder
    print("\n🚀 Testing knowledge graph creation from cleaned blog articles...")
    
    cmd = [
        sys.executable, 
        "scripts/build_knowledge_graph.py",
        "--source-type", "blog",
        "--input-dir", str(cleaned_articles_dir),
        "--version-name", "cleaned_blog_test",
        "--description", "Test build from cleaned blog articles"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        print("\n📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n📥 STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Knowledge graph creation test completed successfully!")
            
            # List versions to verify
            print("\n📋 Listing versions to verify creation:")
            list_cmd = [sys.executable, "scripts/build_knowledge_graph.py", "--list-versions"]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True, cwd=Path.cwd())
            print(list_result.stdout)
            
            return True
        else:
            print(f"\n❌ Knowledge graph creation failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)