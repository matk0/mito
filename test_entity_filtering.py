#!/usr/bin/env python3

"""
Test entity filtering to see what gets removed vs kept.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_entity_filtering():
    """Test the entity filtering logic."""
    
    print("🧪 Testing Entity Filtering")
    print("=" * 50)
    
    try:
        from neo4j_graph_builder import SlovakHealthGraphBuilder
        
        # Load entities
        entities_file = "./chunked_data/extracted_entities.json"
        if not Path(entities_file).exists():
            print(f"❌ Entities file not found: {entities_file}")
            return
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunk_entities = data['chunk_entities']
        
        # Initialize the builder (just for filtering methods)
        builder = SlovakHealthGraphBuilder()
        
        # Collect all entities and test filtering
        all_entities = []
        filtered_entities = []
        removed_entities = []
        
        for chunk_data in chunk_entities:
            for entity in chunk_data['entities']:
                all_entities.append(entity)
                
                # Test filtering
                should_filter = (
                    entity['label'].startswith('SPACY_') or 
                    entity['label'] == 'CHEMICAL_FORMULA' and len(entity['text']) <= 2 or
                    builder._is_common_word(entity['text']) or
                    builder._is_low_quality_entity(entity)
                )
                
                if should_filter:
                    removed_entities.append(entity)
                else:
                    filtered_entities.append(entity)
        
        print(f"📊 Filtering Results:")
        print(f"   Total entities: {len(all_entities)}")
        print(f"   Entities kept: {len(filtered_entities)}")
        print(f"   Entities removed: {len(removed_entities)}")
        print(f"   Reduction: {len(removed_entities)/len(all_entities)*100:.1f}%")
        
        # Show examples of what gets removed
        print(f"\n❌ Examples of Removed Entities:")
        removed_examples = Counter(entity['text'] for entity in removed_entities).most_common(20)
        for text, count in removed_examples:
            print(f"   '{text}' (removed {count} times)")
        
        # Show examples of what gets kept
        print(f"\n✅ Examples of Kept Domain Entities:")
        kept_domain = [e for e in filtered_entities if not e['label'].startswith('SPACY_')]
        kept_examples = Counter(entity['text'] for entity in kept_domain).most_common(15)
        for text, count in kept_examples:
            print(f"   '{text}' (mentioned {count} times)")
        
        # Show entity type distribution after filtering
        print(f"\n📈 Entity Types After Filtering:")
        kept_types = Counter(entity['label'] for entity in filtered_entities)
        for entity_type, count in kept_types.most_common():
            print(f"   {entity_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_entity_filtering()