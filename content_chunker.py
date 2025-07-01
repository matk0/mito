#!/usr/bin/env python3

import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path

class ContentChunker:
    def __init__(self, 
                 input_dir: str = "./scraped_data/articles",
                 output_dir: str = "./chunked_data",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        Initialize the content chunker.
        
        Args:
            input_dir: Directory containing scraped JSON articles
            output_dir: Directory to save chunked content
            chunk_size: Target size for each chunk (in tokens/words)
            chunk_overlap: Overlap between consecutive chunks
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        self.chunks = []
        self.stats = {
            'total_articles': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'articles_processed': []
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple dots, dashes, etc.
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{2,}', '--', text)
        
        # Clean up common Slovak text artifacts
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ])', r'\1 \2', text)  # Ensure space after sentence
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling Slovak punctuation."""
        # Slovak sentence endings
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean up and filter out very short sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from text content.
        
        Args:
            text: The text content to chunk
            metadata: Article metadata (title, url, date, etc.)
            
        Returns:
            List of chunk dictionaries
        """
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_words > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                
                chunk_data = {
                    'text': chunk_text,
                    'word_count': current_size,
                    'chunk_id': len(chunks),
                    'source_url': metadata.get('url', ''),
                    'source_title': metadata.get('title', ''),
                    'source_date': metadata.get('date', ''),
                    'article_excerpt': metadata.get('excerpt', ''),
                    'chunk_start_sentence': current_chunk[0][:100] + '...' if current_chunk[0] else '',
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Calculate how many sentences to keep for overlap
                    overlap_words = 0
                    overlap_sentences = []
                    
                    # Start from the end and work backwards
                    for j in range(len(current_chunk) - 1, -1, -1):
                        sent_words = len(current_chunk[j].split())
                        if overlap_words + sent_words <= self.chunk_overlap:
                            overlap_sentences.insert(0, current_chunk[j])
                            overlap_words += sent_words
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_words
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add current sentence
            current_chunk.append(sentence)
            current_size += sentence_words
            i += 1
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                'text': chunk_text,
                'word_count': current_size,
                'chunk_id': len(chunks),
                'source_url': metadata.get('url', ''),
                'source_title': metadata.get('title', ''),
                'source_date': metadata.get('date', ''),
                'article_excerpt': metadata.get('excerpt', ''),
                'chunk_start_sentence': current_chunk[0][:100] + '...' if current_chunk[0] else '',
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def process_article(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single article file and return chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            # Extract content and metadata
            content = article_data.get('content', '')
            metadata = {
                'title': article_data.get('title', ''),
                'url': article_data.get('url', ''),
                'date': article_data.get('date', ''),
                'excerpt': article_data.get('excerpt', ''),
                'word_count': article_data.get('word_count', 0),
                'scraped_at': article_data.get('scraped_at', ''),
                'filename': file_path.name
            }
            
            if not content or len(content.strip()) < 100:
                print(f"⚠️  Skipping {file_path.name}: content too short")
                return []
            
            chunks = self.create_chunks(content, metadata)
            
            print(f"✅ Processed {file_path.name}: {len(chunks)} chunks created")
            self.stats['articles_processed'].append({
                'filename': file_path.name,
                'title': metadata['title'][:50] + '...' if len(metadata['title']) > 50 else metadata['title'],
                'chunks_created': len(chunks),
                'original_words': metadata['word_count']
            })
            
            return chunks
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            return []
    
    def process_all_articles(self):
        """Process all articles in the input directory."""
        print("🚀 Starting content chunking process...")
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"⚙️  Chunk size: {self.chunk_size} words, Overlap: {self.chunk_overlap} words")
        print("-" * 60)
        
        json_files = list(self.input_dir.glob("*.json"))
        
        if not json_files:
            print("❌ No JSON files found in input directory!")
            return
        
        print(f"📄 Found {len(json_files)} articles to process\n")
        
        all_chunks = []
        
        for i, file_path in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Processing: {file_path.name}")
            
            chunks = self.process_article(file_path)
            if chunks:
                all_chunks.extend(chunks)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"📊 Progress: {i}/{len(json_files)} files processed, {len(all_chunks)} chunks created so far\n")
        
        # Add chunk global IDs
        for i, chunk in enumerate(all_chunks):
            chunk['global_chunk_id'] = i
        
        self.chunks = all_chunks
        
        # Calculate statistics
        self.stats['total_articles'] = len(json_files)
        self.stats['total_chunks'] = len(all_chunks)
        self.stats['avg_chunk_size'] = sum(chunk['word_count'] for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        print("\n" + "=" * 60)
        print("✅ CHUNKING COMPLETE!")
        print(f"📄 Articles processed: {self.stats['total_articles']}")
        print(f"🧩 Total chunks created: {self.stats['total_chunks']}")
        print(f"📏 Average chunk size: {self.stats['avg_chunk_size']:.1f} words")
        print("=" * 60)
    
    def save_chunks(self, filename: str = "chunked_content.json"):
        """Save all chunks to a JSON file."""
        output_file = self.output_dir / filename
        
        data = {
            'chunks': self.chunks,
            'metadata': {
                'total_chunks': len(self.chunks),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'statistics': self.stats
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Chunks saved to: {output_file}")
        
        # Also save statistics separately
        stats_file = self.output_dir / "chunking_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Statistics saved to: {stats_file}")
    
    def preview_chunks(self, num_chunks: int = 3):
        """Preview a few chunks to verify quality."""
        if not self.chunks:
            print("No chunks available to preview!")
            return
        
        print(f"\n🔍 PREVIEW: First {min(num_chunks, len(self.chunks))} chunks:")
        print("-" * 80)
        
        for i in range(min(num_chunks, len(self.chunks))):
            chunk = self.chunks[i]
            print(f"\n📝 Chunk #{i + 1}")
            print(f"📄 Source: {chunk['source_title'][:60]}...")
            print(f"📏 Words: {chunk['word_count']}")
            print(f"🔗 URL: {chunk['source_url']}")
            print(f"📅 Date: {chunk['source_date']}")
            print(f"📖 Content preview:")
            print(f"   {chunk['text'][:200]}...")
            print("-" * 40)


def main():
    """Main function to run the chunking process."""
    print("Slovak Blog Content Chunker")
    print("=" * 40)
    
    # Initialize chunker with optimized settings for Slovak content
    chunker = ContentChunker(
        input_dir="./scraped_data/articles",
        output_dir="./chunked_data",
        chunk_size=800,    # Good balance for Slovak text
        chunk_overlap=100  # Maintain context between chunks
    )
    
    # Process all articles
    chunker.process_all_articles()
    
    # Save results
    chunker.save_chunks()
    
    # Show preview
    chunker.preview_chunks(3)


if __name__ == "__main__":
    main()