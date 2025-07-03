#!/usr/bin/env python3

"""
Test Slovak entity extraction for GraphRAG system.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from entity_extractor import SlovakHealthEntityExtractor

def test_slovak_queries():
    """Test entity extraction on Slovak queries."""
    print("🧪 Testing Slovak Entity Extraction")
    print("=" * 50)
    
    # Initialize extractor
    extractor = SlovakHealthEntityExtractor()
    
    # Test queries
    test_queries = [
        "Ako mitochondrie súvisia s ATP a energiou?",
        "Čo je kvantová biológia a svetlo?", 
        "Ako DHA ovplyvňuje zdravie mitochondrií?",
        "Aký je vplyv chladného šoku na hormóny?",
        "Dopamín a serotonín v mozgu",
        "Vitamín D3 a slnečné svetlo",
        "Kortizol a stres v tele"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Query {i}: {query}")
        
        entities = extractor.extract_entities(query)
        
        if entities:
            print(f"   🏷️  Found {len(entities)} entities:")
            for entity in entities:
                print(f"      - {entity.text} ({entity.label})")
        else:
            print(f"   ❌ No entities found")
    
    print("\n" + "=" * 50)
    print("✅ Testing complete")

if __name__ == "__main__":
    test_slovak_queries()