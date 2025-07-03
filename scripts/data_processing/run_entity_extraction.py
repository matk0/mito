#!/usr/bin/env python3

"""
Optimized script to run entity extraction on all chunked content with progress tracking.
"""

import json
import sys
from pathlib import Path
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def run_entity_extraction_optimized():
    """Run entity extraction on all chunks with optimizations."""
    
    print("🚀 Running Entity Extraction on All Chunks")
    print("=" * 60)
    
    try:
        from entity_extractor import SlovakHealthEntityExtractor
        
        # Initialize extractor
        print("🔧 Initializing entity extractor...")
        extractor = SlovakHealthEntityExtractor()
        
        # Load chunked data
        chunked_file = "./chunked_data/chunked_content.json"
        if not Path(chunked_file).exists():
            print(f"❌ Chunked content file not found: {chunked_file}")
            return False
        
        with open(chunked_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        total_chunks = len(chunks)
        
        print(f"📄 Processing {total_chunks} chunks...")
        
        # Process chunks in batches for better performance
        batch_size = 50
        all_chunk_entities = []
        all_entities = []
        
        start_time = time.time()
        
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch = chunks[batch_start:batch_end]
            
            print(f"📦 Processing batch {batch_start//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            print(f"   Chunks {batch_start + 1}-{batch_end} of {total_chunks}")
            
            batch_start_time = time.time()
            
            for i, chunk in enumerate(batch):
                chunk_idx = batch_start + i
                
                # Extract entities from main text
                entities = extractor.extract_entities(chunk['text'])
                
                # Add context entities if available (but limit to avoid duplication)
                context_entities = []
                if chunk.get('preceding_context') and len(chunk['preceding_context']) > 50:
                    context_entities.extend(extractor.extract_entities(chunk['preceding_context'][:500]))
                
                if chunk.get('following_context') and len(chunk['following_context']) > 50:
                    context_entities.extend(extractor.extract_entities(chunk['following_context'][:500]))
                
                # Remove duplicates and combine
                all_chunk_entities_text = [e.text.lower() for e in entities]
                for ce in context_entities:
                    if ce.text.lower() not in all_chunk_entities_text:
                        entities.append(ce)
                
                # Create chunk entity data
                chunk_entity_data = {
                    'chunk_id': chunk.get('global_chunk_id', chunk_idx),
                    'source_title': chunk.get('source_title', ''),
                    'source_url': chunk.get('source_url', ''),
                    'source_date': chunk.get('source_date', ''),
                    'chunk_position': chunk.get('chunk_position', ''),
                    'entities': [
                        {
                            'text': entity.text,
                            'label': entity.label,
                            'start': entity.start,
                            'end': entity.end,
                            'confidence': entity.confidence
                        }
                        for entity in entities
                    ]
                }
                
                all_chunk_entities.append(chunk_entity_data)
                all_entities.extend(entities)
            
            batch_time = time.time() - batch_start_time
            entities_in_batch = sum(len(ce['entities']) for ce in all_chunk_entities[batch_start:batch_end])
            
            print(f"   ✅ Batch completed in {batch_time:.1f}s")
            print(f"   📊 {entities_in_batch} entities extracted from {len(batch)} chunks")
            
            # Save intermediate results every 5 batches
            if (batch_start // batch_size + 1) % 5 == 0:
                _save_intermediate_results(all_chunk_entities, batch_start // batch_size + 1)
        
        # Generate final statistics
        entity_stats = _generate_entity_statistics(all_entities)
        
        # Create final results
        result = {
            'chunk_entities': all_chunk_entities,
            'extraction_metadata': {
                'total_chunks_processed': total_chunks,
                'total_entities_extracted': len(all_entities),
                'entity_type_counts': dict(extractor.extraction_stats),
                'entity_statistics': entity_stats,
                'extraction_timestamp': __import__('datetime').datetime.now().isoformat(),
                'processing_time_seconds': time.time() - start_time
            }
        }
        
        # Save final results
        output_file = "./chunked_data/extracted_entities.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ ENTITY EXTRACTION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📄 Chunks processed: {total_chunks}")
        print(f"🏷️  Total entities: {len(all_entities)}")
        print(f"📈 Entity types: {len(extractor.extraction_stats)}")
        print(f"💾 Results saved to: {output_file}")
        
        # Show top entity types
        print(f"\n🔝 Top 10 Entity Types:")
        for entity_type, count in sorted(extractor.extraction_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {entity_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def _save_intermediate_results(chunk_entities, batch_num):
    """Save intermediate results for recovery."""
    try:
        intermediate_file = f"./chunked_data/extracted_entities_batch_{batch_num}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump({'chunk_entities': chunk_entities}, f, ensure_ascii=False, indent=2)
        print(f"   💾 Intermediate results saved: batch {batch_num}")
    except Exception as e:
        print(f"   ⚠️  Could not save intermediate results: {e}")

def _generate_entity_statistics(entities):
    """Generate comprehensive statistics about extracted entities."""
    from collections import defaultdict
    
    stats = {
        'total_entities': len(entities),
        'unique_entities': len(set(entity.text.lower() for entity in entities)),
        'entity_types': len(set(entity.label for entity in entities)),
        'avg_confidence': sum(entity.confidence for entity in entities) / len(entities) if entities else 0,
        'top_entities': {},
        'top_entity_types': {}
    }
    
    # Count entity frequencies
    entity_counts = defaultdict(int)
    for entity in entities:
        entity_counts[entity.text.lower()] += 1
    
    # Top 20 most frequent entities
    stats['top_entities'] = dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    # Top entity types
    type_counts = defaultdict(int)
    for entity in entities:
        type_counts[entity.label] += 1
    
    stats['top_entity_types'] = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
    
    return stats

if __name__ == "__main__":
    success = run_entity_extraction_optimized()
    if success:
        print(f"\n🎉 Entity extraction completed successfully!")
    else:
        print(f"\n💥 Entity extraction failed!")