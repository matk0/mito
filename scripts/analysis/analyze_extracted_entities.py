#!/usr/bin/env python3

"""
Analyze the extracted entities and generate insights for GraphRAG construction.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import re

def analyze_extracted_entities():
    """Analyze the extracted entities and generate insights."""
    
    print("📊 Analyzing Extracted Entities")
    print("=" * 50)
    
    # Load extracted entities
    entities_file = "./chunked_data/extracted_entities.json"
    if not Path(entities_file).exists():
        print(f"❌ Entities file not found: {entities_file}")
        return
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunk_entities = data['chunk_entities']
    metadata = data['extraction_metadata']
    
    print(f"📄 Processed {metadata['total_chunks_processed']} chunks")
    print(f"🏷️  Extracted {metadata['total_entities_extracted']} total entities")
    print(f"📈 Found {len(metadata['entity_type_counts'])} entity types")
    
    # Analyze domain-specific entities (filter out generic spaCy entities)
    domain_entities = {}
    spacy_entities = {}
    
    for chunk in chunk_entities:
        for entity in chunk['entities']:
            label = entity['label']
            text = entity['text'].strip()
            
            if label.startswith('SPACY_'):
                if label not in spacy_entities:
                    spacy_entities[label] = []
                spacy_entities[label].append(text)
            else:
                if label not in domain_entities:
                    domain_entities[label] = []
                domain_entities[label].append(text)
    
    print(f"\n🧬 Domain-Specific Entities Analysis:")
    print("=" * 50)
    
    domain_stats = {}
    for entity_type, entities in domain_entities.items():
        unique_entities = list(set(entities))
        domain_stats[entity_type] = {
            'total_mentions': len(entities),
            'unique_entities': len(unique_entities),
            'top_entities': Counter(entities).most_common(10)
        }
        
        print(f"\n📝 {entity_type}:")
        print(f"   Total mentions: {len(entities)}")
        print(f"   Unique entities: {len(unique_entities)}")
        print(f"   Top entities: {[item[0] for item in Counter(entities).most_common(5)]}")
    
    # Analyze co-occurrence patterns
    print(f"\n🔗 Entity Co-occurrence Analysis:")
    print("=" * 50)
    
    co_occurrence = defaultdict(lambda: defaultdict(int))
    entity_chunk_map = defaultdict(set)
    
    for chunk in chunk_entities:
        chunk_id = chunk['chunk_id']
        chunk_entities_text = []
        
        for entity in chunk['entities']:
            if not entity['label'].startswith('SPACY_') and not entity['label'] == 'CHEMICAL_FORMULA':
                entity_text = entity['text'].lower().strip()
                if len(entity_text) > 2:  # Filter very short entities
                    chunk_entities_text.append(entity_text)
                    entity_chunk_map[entity_text].add(chunk_id)
        
        # Calculate co-occurrences within chunk
        for i, entity1 in enumerate(chunk_entities_text):
            for entity2 in chunk_entities_text[i+1:]:
                if entity1 != entity2:
                    co_occurrence[entity1][entity2] += 1
                    co_occurrence[entity2][entity1] += 1
    
    # Find most connected entities
    entity_connections = {}
    for entity, connections in co_occurrence.items():
        entity_connections[entity] = {
            'total_connections': sum(connections.values()),
            'unique_connections': len(connections),
            'top_connected': sorted(connections.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    # Top 10 most connected entities
    top_connected = sorted(entity_connections.items(), 
                          key=lambda x: x[1]['total_connections'], 
                          reverse=True)[:10]
    
    print("🌐 Most Connected Entities:")
    for entity, stats in top_connected:
        print(f"   {entity}: {stats['total_connections']} connections")
        top_connected_entities = [f"{conn[0]} ({conn[1]})" for conn in stats['top_connected']]
        print(f"      Connected to: {', '.join(top_connected_entities)}")
    
    # Analyze entity distribution across articles
    print(f"\n📚 Entity Distribution Analysis:")
    print("=" * 50)
    
    article_entities = defaultdict(lambda: defaultdict(int))
    
    for chunk in chunk_entities:
        article_title = chunk['source_title']
        for entity in chunk['entities']:
            if not entity['label'].startswith('SPACY_') and entity['label'] != 'CHEMICAL_FORMULA':
                entity_text = entity['text'].lower().strip()
                if len(entity_text) > 2:
                    article_entities[article_title][entity_text] += 1
    
    # Find articles with most diverse entities
    article_diversity = {}
    for article, entities in article_entities.items():
        article_diversity[article] = {
            'unique_entities': len(entities),
            'total_mentions': sum(entities.values()),
            'top_entities': sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    top_diverse_articles = sorted(article_diversity.items(), 
                                 key=lambda x: x[1]['unique_entities'], 
                                 reverse=True)[:5]
    
    print("📖 Most Entity-Rich Articles:")
    for article, stats in top_diverse_articles:
        print(f"   {article[:60]}...")
        print(f"      Unique entities: {stats['unique_entities']}")
        print(f"      Total mentions: {stats['total_mentions']}")
        top_entities = [f"{ent[0]} ({ent[1]})" for ent in stats['top_entities']]
        print(f"      Top entities: {', '.join(top_entities)}")
    
    # Generate recommendations for GraphRAG
    print(f"\n🎯 GraphRAG Construction Recommendations:")
    print("=" * 50)
    
    print("1. 🏗️ Node Types to Create:")
    for entity_type, stats in domain_stats.items():
        if stats['unique_entities'] > 5 and stats['total_mentions'] > 50:
            print(f"   • {entity_type}: {stats['unique_entities']} unique entities")
    
    print("\n2. 🔗 Key Relationship Patterns:")
    print("   • High co-occurrence entities suggest strong relationships")
    print("   • Cross-article entity mentions indicate global concepts")
    print("   • Entity density per article shows concept clustering")
    
    print("\n3. 📊 Quality Metrics:")
    meaningful_entities = sum(stats['unique_entities'] for stats in domain_stats.values())
    print(f"   • Meaningful entities: {meaningful_entities}")
    print(f"   • Average connections per entity: {sum(stats['total_connections'] for stats in entity_connections.values()) / len(entity_connections) if entity_connections else 0:.1f}")
    print(f"   • Articles with rich entity content: {len([a for a in article_diversity.values() if a['unique_entities'] > 20])}")
    
    # Save analysis results
    analysis_result = {
        'domain_entity_stats': domain_stats,
        'top_connected_entities': dict(top_connected),
        'article_diversity': dict(top_diverse_articles),
        'entity_co_occurrence_sample': dict(list(co_occurrence.items())[:10]),
        'recommendations': {
            'viable_node_types': [et for et, stats in domain_stats.items() 
                                 if stats['unique_entities'] > 5 and stats['total_mentions'] > 50],
            'most_connected_entities': [entity for entity, _ in top_connected],
            'richest_articles': [article for article, _ in top_diverse_articles]
        }
    }
    
    output_file = "./chunked_data/entity_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_extracted_entities()