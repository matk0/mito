#!/usr/bin/env python3

"""
Test entity extraction on a small subset of chunked content.
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_entity_extraction_on_sample_chunks():
    """Test entity extraction on a small sample of actual chunks."""
    
    print("🧪 Testing Entity Extraction on Sample Chunks")
    print("=" * 60)
    
    try:
        from entity_extractor import SlovakHealthEntityExtractor
        
        # Load chunked data
        chunked_file = "./chunked_data/chunked_content.json"
        if not Path(chunked_file).exists():
            print(f"❌ Chunked content file not found: {chunked_file}")
            return False
        
        with open(chunked_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        total_chunks = len(chunks)
        
        print(f"📄 Found {total_chunks} total chunks")
        
        # Take only first 5 chunks for testing
        sample_chunks = chunks[:5]
        print(f"🔍 Testing on first {len(sample_chunks)} chunks")
        
        # Initialize extractor
        extractor = SlovakHealthEntityExtractor()
        
        # Process sample chunks
        all_entities = []
        
        for i, chunk in enumerate(sample_chunks):
            print(f"\n📝 Processing chunk {i+1}/{len(sample_chunks)}")
            print(f"   Title: {chunk.get('source_title', '')[:50]}...")
            print(f"   Text length: {len(chunk['text'])} characters")
            
            # Extract entities
            entities = extractor.extract_entities(chunk['text'])
            all_entities.extend(entities)
            
            print(f"   Entities found: {len(entities)}")
            
            # Show top entities for this chunk
            entity_types = {}
            for entity in entities:
                if entity.label not in entity_types:
                    entity_types[entity.label] = []
                entity_types[entity.label].append(entity.text)
            
            for entity_type, entity_texts in list(entity_types.items())[:3]:  # Show top 3 types
                unique_texts = list(set(entity_texts))[:3]  # Show up to 3 unique entities
                print(f"     {entity_type}: {', '.join(unique_texts)}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total entities extracted: {len(all_entities)}")
        print(f"   Unique entities: {len(set(e.text for e in all_entities))}")
        print(f"   Entity types found: {len(set(e.label for e in all_entities))}")
        
        # Most common entity types
        type_counts = {}
        for entity in all_entities:
            type_counts[entity.label] = type_counts.get(entity.label, 0) + 1
        
        print(f"\n🔝 Top Entity Types:")
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {entity_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_entity_extraction_on_sample_chunks()
    if success:
        print(f"\n✅ Sample entity extraction test completed successfully!")
    else:
        print(f"\n❌ Sample entity extraction test failed!")