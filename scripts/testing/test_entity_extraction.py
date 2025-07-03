#!/usr/bin/env python3

"""
Test script for the Slovak Health Entity Extractor.
Tests basic functionality without requiring the full chunked dataset.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_entity_extraction():
    """Test entity extraction with sample Slovak health text."""
    
    # Sample Slovak health text from the articles
    sample_text = """
    Mitochondrie sú kľúčové pre tvorbu ATP a energia bunky. 
    Dopamín a serotonín ovplyvňujú naše zdravie cez epigenetiku.
    UV svetlo a infračervené svetlo podporujú kvantovú biológiu.
    Vitamín D3 a DHA sú dôležité pre mitochondriálnu funkciu.
    Deutérium a štruktúrovaná voda majú vplyv na buničné procesy.
    Kortizol a leptín regulujú metabolizmus cez cirkadiálny rytmus.
    """
    
    print("🧪 Testing Slovak Health Entity Extractor")
    print("=" * 50)
    
    try:
        from entity_extractor import SlovakHealthEntityExtractor
        
        # Initialize extractor
        print("🚀 Initializing entity extractor...")
        extractor = SlovakHealthEntityExtractor()
        
        print("🔍 Extracting entities from sample text...")
        entities = extractor.extract_entities(sample_text)
        
        print(f"✅ Extracted {len(entities)} entities:")
        print("-" * 30)
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.label not in entities_by_type:
                entities_by_type[entity.label] = []
            entities_by_type[entity.label].append(entity.text)
        
        # Display results
        for entity_type, entity_texts in entities_by_type.items():
            print(f"\n📝 {entity_type}:")
            for text in set(entity_texts):  # Remove duplicates
                print(f"   • {text}")
        
        print(f"\n📊 Summary:")
        print(f"   Total entities: {len(entities)}")
        print(f"   Unique entity types: {len(entities_by_type)}")
        print(f"   Unique entities: {len(set(e.text for e in entities))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing entity extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_taxonomy_coverage():
    """Test the coverage of our entity taxonomy."""
    print("\n🏷️  Testing Entity Taxonomy Coverage")
    print("=" * 50)
    
    try:
        from entity_extractor import SlovakHealthEntityExtractor
        
        extractor = SlovakHealthEntityExtractor()
        
        print(f"📚 Entity taxonomy contains {len(extractor.entity_taxonomy)} categories:")
        
        for category, terms in extractor.entity_taxonomy.items():
            print(f"   • {category}: {len(terms)} terms")
        
        total_terms = sum(len(terms) for terms in extractor.entity_taxonomy.values())
        print(f"\n📊 Total terms in taxonomy: {total_terms}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing taxonomy: {e}")
        return False

if __name__ == "__main__":
    print("Slovak Health Entity Extractor - Test Suite")
    print("=" * 60)
    
    # Test 1: Basic entity extraction
    test1_success = test_entity_extraction()
    
    # Test 2: Taxonomy coverage
    test2_success = test_taxonomy_coverage()
    
    # Overall result
    print("\n" + "=" * 60)
    if test1_success and test2_success:
        print("✅ All tests passed! Entity extractor is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")