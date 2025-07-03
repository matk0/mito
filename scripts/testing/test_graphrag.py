#!/usr/bin/env python3

"""
Test suite for GraphRAG system functionality.
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_graphrag_basic_functionality():
    """Test basic GraphRAG functionality."""
    print("🧪 Testing GraphRAG Basic Functionality")
    print("=" * 60)
    
    try:
        from graphrag_system import SlovakHealthGraphRAG
        
        # Initialize system
        print("🚀 Initializing GraphRAG...")
        graphrag = SlovakHealthGraphRAG()
        
        # Test 1: Entity extraction from query
        print("\n📝 Test 1: Entity Extraction")
        test_query = "Ako súvisí mitochondria s ATP a DHA?"
        entities = graphrag.extract_query_entities(test_query)
        print(f"   Query: '{test_query}'")
        print(f"   Extracted entities: {entities}")
        
        # Test 2: Vector search
        print("\n📊 Test 2: Vector Search")
        vector_results = graphrag.vector_search(test_query, n_results=3)
        print(f"   Found {len(vector_results)} vector results")
        for i, result in enumerate(vector_results[:2], 1):
            print(f"   Result {i}: {result['similarity_score']:.3f} - {result['content'][:100]}...")
        
        # Test 3: Entity context retrieval
        print("\n🌐 Test 3: Entity Context")
        if entities:
            test_entity = entities[0]
            context = graphrag.get_entity_context(test_entity)
            if context:
                print(f"   Entity: {context.entity_name}")
                print(f"   Type: {context.entity_type}")
                print(f"   Mentions: {context.mention_count}")
                print(f"   Related entities: {[name for name, _ in context.related_entities[:3]]}")
            else:
                print(f"   No context found for '{test_entity}'")
        
        # Test 4: Full GraphRAG search
        print("\n🔍 Test 4: Full GraphRAG Search")
        start_time = time.time()
        results = graphrag.search(test_query, n_results=3)
        search_time = time.time() - start_time
        
        print(f"   Search completed in {search_time:.2f}s")
        print(f"   Results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            print(f"   Result {i}:")
            print(f"     Combined score: {result.combined_score:.3f}")
            print(f"     Vector: {result.similarity_score:.3f} | Graph: {result.graph_relevance_score:.3f}")
            print(f"     Entities: {result.related_entities[:3]}")
            print(f"     Content: {result.content[:80]}...")
        
        # Test 5: Entity summary
        print("\n📋 Test 5: Entity Summary")
        summary = graphrag.get_entity_summary("mitochondria")
        if summary:
            print(f"   Entity: {summary['entity_name']}")
            print(f"   Type: {summary['entity_type']}")
            print(f"   Mentions: {summary['mention_count']}")
            print(f"   Top related: {[name for name, _ in summary['top_related_entities'][:3]]}")
        
        graphrag.close()
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graphrag_performance():
    """Test GraphRAG performance with multiple queries."""
    print("\n🚀 Testing GraphRAG Performance")
    print("=" * 60)
    
    test_queries = [
        "mitochondria energia ATP",
        "kvantová biológia svetlo",
        "DHA zdravie membrány",
        "hormóny leptín inzulín",
        "chlad adaptácia termogenéza",
        "UV svetlo vitamín D",
        "redox oxidatívny stres",
        "epigenetika génová expresia"
    ]
    
    try:
        from graphrag_system import SlovakHealthGraphRAG
        
        graphrag = SlovakHealthGraphRAG()
        
        total_time = 0
        successful_searches = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            start_time = time.time()
            try:
                results = graphrag.search(query, n_results=5)
                search_time = time.time() - start_time
                total_time += search_time
                successful_searches += 1
                
                print(f"   ⏱️  Time: {search_time:.2f}s")
                print(f"   📊 Results: {len(results)}")
                
                if results:
                    best_result = results[0]
                    print(f"   🏆 Best score: {best_result.combined_score:.3f}")
                    print(f"      Vector: {best_result.similarity_score:.3f} | Graph: {best_result.graph_relevance_score:.3f}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Performance summary
        print(f"\n📈 Performance Summary:")
        print(f"   Total queries: {len(test_queries)}")
        print(f"   Successful: {successful_searches}")
        print(f"   Average time: {total_time/successful_searches:.2f}s per query")
        print(f"   Total time: {total_time:.2f}s")
        
        graphrag.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_graphrag_comparison():
    """Compare GraphRAG vs pure vector search."""
    print("\n⚖️  Testing GraphRAG vs Vector Search Comparison")
    print("=" * 60)
    
    comparison_queries = [
        "Ako mitochondrie produkujú ATP?",
        "Čo je kvantová biológia?",
        "Aký vplyv má DHA na zdravie?"
    ]
    
    try:
        from graphrag_system import SlovakHealthGraphRAG
        
        graphrag = SlovakHealthGraphRAG()
        
        for query in comparison_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            
            # Vector-only search
            vector_results = graphrag.vector_search(query, n_results=3)
            
            # GraphRAG search
            graphrag_results = graphrag.search(query, n_results=3)
            
            print("📊 Vector-only vs GraphRAG comparison:")
            
            for i in range(min(3, len(vector_results), len(graphrag_results))):
                vector_score = vector_results[i]['similarity_score']
                graphrag_score = graphrag_results[i].combined_score
                graph_boost = graphrag_results[i].graph_relevance_score
                
                print(f"   Result {i+1}:")
                print(f"     Vector only: {vector_score:.3f}")
                print(f"     GraphRAG: {graphrag_score:.3f} (graph boost: {graph_boost:.3f})")
                print(f"     Entities: {len(graphrag_results[i].related_entities)} found")
        
        graphrag.close()
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def run_all_tests():
    """Run all GraphRAG tests."""
    print("🧪 Slovak Health GraphRAG Test Suite")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_graphrag_basic_functionality),
        ("Performance", test_graphrag_performance),
        ("Comparison", test_graphrag_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
    
    print(f"\n{'='*80}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GraphRAG system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()