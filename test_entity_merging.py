#!/usr/bin/env python3

"""
Test the improved entity merging logic for Slovak linguistic variants.
"""

import sys
from pathlib import Path
from collections import Counter

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_slovak_merging():
    """Test the Slovak entity merging improvements."""
    
    print("🧪 Testing Slovak Entity Merging")
    print("=" * 50)
    
    try:
        from neo4j_graph_builder import SlovakHealthGraphBuilder
        
        # Initialize builder (just for testing methods)
        builder = SlovakHealthGraphBuilder()
        
        # Test cases for Slovak normalization
        test_cases = [
            # Mitochondria variants (should all merge to 'mitochondria')
            'mitochondrie',     # Slovak plural
            'mitochondria',     # Standard form
            'Mitochondrie',     # Capitalized Slovak plural
            'Mitochondria',     # Capitalized standard
            'MITOCHONDRIE',     # All caps Slovak
            'MITOCHONDRIA',     # All caps standard
            
            # Other test cases
            'ATP',
            'atp',
            'DHA',
            'dha',
            'UV svetlo',
            'uv svetlo',
            'REDOX',
            'redox',
            'Redox'
        ]
        
        print("🔤 Testing Normalization:")
        normalization_results = {}
        for test_case in test_cases:
            normalized = builder.normalize_entity_name(test_case)
            normalization_results[test_case] = normalized
            print(f"   '{test_case}' → '{normalized}'")
        
        # Test entity merging with sample data
        print(f"\n🔄 Testing Entity Merging:")
        
        sample_entities = [
            {'text': 'mitochondrie', 'label': 'CELLULAR_COMPONENT', 'confidence': 1.0},
            {'text': 'mitochondria', 'label': 'CELLULAR_COMPONENT', 'confidence': 1.0},
            {'text': 'Mitochondrie', 'label': 'CELLULAR_COMPONENT', 'confidence': 0.9},
            {'text': 'MITOCHONDRIE', 'label': 'CELLULAR_COMPONENT', 'confidence': 0.8},
            {'text': 'ATP', 'label': 'TECHNICAL_TERM', 'confidence': 1.0},
            {'text': 'atp', 'label': 'TECHNICAL_TERM', 'confidence': 0.9},
            {'text': 'DHA', 'label': 'FATTY_ACID', 'confidence': 1.0},
            {'text': 'dha', 'label': 'FATTY_ACID', 'confidence': 0.8},
        ]
        
        print(f"📋 Before merging: {len(sample_entities)} entities")
        for entity in sample_entities:
            print(f"   {entity['text']} ({entity['label']})")
        
        merged_entities = builder.merge_similar_entities(sample_entities)
        
        print(f"\n✅ After merging: {len(merged_entities)} entities")
        for entity in merged_entities:
            variants = entity.get('variants', [])
            merged_from = entity.get('merged_from', 1)
            print(f"   {entity['text']} ({entity['label']})")
            print(f"      Normalized: {entity.get('normalized_name', 'N/A')}")
            if merged_from > 1:
                print(f"      Merged from {merged_from} variants: {variants}")
            print()
        
        # Show the impact
        print(f"📊 Merging Impact:")
        print(f"   Original entities: {len(sample_entities)}")
        print(f"   Merged entities: {len(merged_entities)}")
        print(f"   Reduction: {len(sample_entities) - len(merged_entities)} entities")
        
        # Test specific Slovak cases
        print(f"\n🔍 Key Slovak Merging Results:")
        mito_variants = [e for e in merged_entities if 'mitochondr' in e.get('normalized_name', '')]
        if mito_variants:
            mito_entity = mito_variants[0]
            print(f"   Mitochondria merged entity:")
            print(f"      Chosen form: '{mito_entity['text']}'")
            print(f"      All variants: {mito_entity.get('variants', [])}")
            print(f"      Merged from: {mito_entity.get('merged_from', 1)} original entities")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_slovak_merging()
    if success:
        print(f"\n✅ Slovak entity merging test completed!")
        print(f"\n💡 Now rebuild your Neo4j graph to see 'mitochondrie' and 'mitochondria' as one node!")
    else:
        print(f"\n❌ Slovak entity merging test failed!")