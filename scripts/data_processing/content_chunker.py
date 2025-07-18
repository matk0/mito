#!/usr/bin/env python3

import json
import os
import re
from typing import List, Dict, Any, Literal, Optional
from pathlib import Path
from semantic_chunker import SemanticChunker

class ContentChunker:
    def __init__(self, 
                 input_dirs: List[str] = ["./data/raw/scraped_data/articles", "./data/raw/scraped_data/pdfs"],
                 output_dir: str = "./data/processed/chunked_data",
                 chunk_size: int = 800,
                 chunk_overlap: int = 200,
                 context_window: int = 300,
                 chunking_strategy: Literal["fixed", "semantic"] = "fixed",
                 semantic_breakpoint_type: Literal["percentile", "standard_deviation", "gradient"] = "percentile",
                 semantic_breakpoint_amount: float = 85.0):
        """
        Initialize the content chunker.
        
        Args:
            input_dirs: List of directories containing scraped JSON articles
            output_dir: Directory to save chunked content
            chunk_size: Target size for each chunk (in tokens/words) - for fixed strategy
            chunk_overlap: Overlap between consecutive chunks - for fixed strategy
            context_window: Additional context words to include before/after each chunk
            chunking_strategy: Strategy to use: "fixed" or "semantic"
            semantic_breakpoint_type: For semantic chunking: "percentile", "standard_deviation", or "gradient"
            semantic_breakpoint_amount: Threshold value for semantic breakpoints
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window
        self.chunking_strategy = chunking_strategy
        
        # Initialize semantic chunker if needed
        if chunking_strategy == "semantic":
            self.semantic_chunker = SemanticChunker(
                breakpoint_threshold_type=semantic_breakpoint_type,
                breakpoint_threshold_amount=semantic_breakpoint_amount,
                min_chunk_size=200,  # Reasonable minimum for Slovak content
                max_chunk_size=chunk_size,  # Use provided max size
                buffer_size=1
            )
        else:
            self.semantic_chunker = None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        self.chunks = []
        self.stats = {
            'total_articles': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'articles_processed': [],
            'chunking_strategy': chunking_strategy,
            'chunking_params': {
                'strategy': chunking_strategy,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap if chunking_strategy == "fixed" else None,
                'semantic_breakpoint_type': semantic_breakpoint_type if chunking_strategy == "semantic" else None,
                'semantic_breakpoint_amount': semantic_breakpoint_amount if chunking_strategy == "semantic" else None
            }
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
    
    def get_contextual_window(self, sentences: List[str], start_idx: int, end_idx: int) -> Dict[str, str]:
        """
        Extract contextual window around a chunk.
        
        Args:
            sentences: All sentences from the article
            start_idx: Starting sentence index of the chunk
            end_idx: Ending sentence index of the chunk
            
        Returns:
            Dictionary with preceding and following context
        """
        # Calculate context boundaries
        context_start = max(0, start_idx - 1)
        context_end = min(len(sentences), end_idx + 1)
        
        # Get preceding context
        preceding_context = []
        preceding_words = 0
        for i in range(start_idx - 1, -1, -1):
            sent_words = len(sentences[i].split())
            if preceding_words + sent_words <= self.context_window:
                preceding_context.insert(0, sentences[i])
                preceding_words += sent_words
            else:
                break
        
        # Get following context
        following_context = []
        following_words = 0
        for i in range(end_idx, len(sentences)):
            sent_words = len(sentences[i].split())
            if following_words + sent_words <= self.context_window:
                following_context.append(sentences[i])
                following_words += sent_words
            else:
                break
        
        return {
            'preceding_context': ' '.join(preceding_context),
            'following_context': ' '.join(following_context),
            'preceding_word_count': preceding_words,
            'following_word_count': following_words
        }

    def extract_section_headers(self, text: str) -> List[str]:
        """Extract potential section headers from text."""
        lines = text.split('\n')
        headers = []
        
        for line in lines:
            line = line.strip()
            # Check for patterns that might be headers
            if (len(line) > 0 and len(line) < 100 and 
                (line.isupper() or 
                 line.count(':') == 1 and line.endswith(':') or
                 line.startswith('#') or
                 len(line.split()) <= 8)):
                headers.append(line)
        
        return headers[:5]  # Limit to first 5 potential headers

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from text content with enhanced context and metadata.
        
        Args:
            text: The text content to chunk
            metadata: Article metadata (title, url, date, etc.)
            
        Returns:
            List of chunk dictionaries with enhanced metadata
        """
        cleaned_text = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned_text)
        section_headers = self.extract_section_headers(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start_idx = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_words > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk_end_idx = i
                
                # Get contextual window
                context = self.get_contextual_window(sentences, chunk_start_idx, chunk_end_idx)
                
                # Determine chunk position in article
                chunk_position = "beginning" if len(chunks) == 0 else "middle"
                if chunk_end_idx >= len(sentences) - 2:
                    chunk_position = "end"
                
                chunk_data = {
                    'text': chunk_text,
                    'word_count': current_size,
                    'chunk_id': len(chunks),
                    
                    # Enhanced source attribution
                    'source_url': metadata.get('url', ''),
                    'source_title': metadata.get('title', ''),
                    'source_date': metadata.get('date', ''),
                    'source_scraped_at': metadata.get('scraped_at', ''),
                    'source_filename': metadata.get('filename', ''),
                    'article_excerpt': metadata.get('excerpt', ''),
                    'article_word_count': metadata.get('word_count', 0),
                    
                    # Contextual information
                    'preceding_context': context['preceding_context'],
                    'following_context': context['following_context'],
                    'preceding_context_words': context['preceding_word_count'],
                    'following_context_words': context['following_word_count'],
                    
                    # Chunk positioning
                    'chunk_position': chunk_position,
                    'chunk_start_sentence': current_chunk[0][:150] + '...' if current_chunk[0] else '',
                    'chunk_end_sentence': current_chunk[-1][:150] + '...' if current_chunk[-1] else '',
                    'sentence_start_idx': chunk_start_idx,
                    'sentence_end_idx': chunk_end_idx - 1,
                    
                    # Article structure context
                    'article_section_headers': section_headers,
                    'total_sentences_in_article': len(sentences),
                    'chunk_sentence_count': len(current_chunk),
                    
                    # For potential adjacent chunk retrieval
                    'adjacent_chunk_ids': {
                        'previous': len(chunks) - 1 if len(chunks) > 0 else None,
                        'next': len(chunks) + 1  # Will be updated if next chunk exists
                    }
                }
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Calculate how many sentences to keep for overlap
                    overlap_words = 0
                    overlap_sentences = []
                    overlap_start_idx = chunk_end_idx
                    
                    # Start from the end and work backwards
                    for j in range(len(current_chunk) - 1, -1, -1):
                        sent_words = len(current_chunk[j].split())
                        if overlap_words + sent_words <= self.chunk_overlap:
                            overlap_sentences.insert(0, current_chunk[j])
                            overlap_words += sent_words
                            overlap_start_idx = chunk_start_idx + j
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_words
                    chunk_start_idx = overlap_start_idx
                else:
                    current_chunk = []
                    current_size = 0
                    chunk_start_idx = i
            
            # Add current sentence
            current_chunk.append(sentence)
            current_size += sentence_words
            i += 1
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_end_idx = len(sentences)
            
            # Get contextual window for last chunk
            context = self.get_contextual_window(sentences, chunk_start_idx, chunk_end_idx)
            
            chunk_data = {
                'text': chunk_text,
                'word_count': current_size,
                'chunk_id': len(chunks),
                
                # Enhanced source attribution
                'source_url': metadata.get('url', ''),
                'source_title': metadata.get('title', ''),
                'source_date': metadata.get('date', ''),
                'source_scraped_at': metadata.get('scraped_at', ''),
                'source_filename': metadata.get('filename', ''),
                'article_excerpt': metadata.get('excerpt', ''),
                'article_word_count': metadata.get('word_count', 0),
                
                # Contextual information
                'preceding_context': context['preceding_context'],
                'following_context': context['following_context'],
                'preceding_context_words': context['preceding_word_count'],
                'following_context_words': context['following_word_count'],
                
                # Chunk positioning
                'chunk_position': 'end' if len(chunks) > 0 else 'complete',
                'chunk_start_sentence': current_chunk[0][:150] + '...' if current_chunk[0] else '',
                'chunk_end_sentence': current_chunk[-1][:150] + '...' if current_chunk[-1] else '',
                'sentence_start_idx': chunk_start_idx,
                'sentence_end_idx': chunk_end_idx - 1,
                
                # Article structure context
                'article_section_headers': section_headers,
                'total_sentences_in_article': len(sentences),
                'chunk_sentence_count': len(current_chunk),
                
                # For potential adjacent chunk retrieval
                'adjacent_chunk_ids': {
                    'previous': len(chunks) - 1 if len(chunks) > 0 else None,
                    'next': None
                }
            }
            chunks.append(chunk_data)
        
        # Update next chunk IDs for all chunks except the last
        for i, chunk in enumerate(chunks[:-1]):
            if chunk['adjacent_chunk_ids']['next'] == i + 1:
                chunk['adjacent_chunk_ids']['next'] = i + 1
        
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
            
            # Choose chunking method based on strategy
            if self.chunking_strategy == "semantic":
                chunks = self.semantic_chunker.chunk_with_metadata(content, metadata)
            else:
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
        """Process all articles in the input directories."""
        print("🚀 Starting enhanced content chunking process...")
        print(f"📂 Input directories: {[str(d) for d in self.input_dirs]}")
        print(f"📂 Output directory: {self.output_dir}")
        print(f"🔧 Chunking strategy: {self.chunking_strategy}")
        if self.chunking_strategy == "fixed":
            print(f"⚙️  Chunk size: {self.chunk_size} words, Overlap: {self.chunk_overlap} words")
            print(f"🔍 Context window: {self.context_window} words (preceding + following)")
        else:
            print(f"⚙️  Max chunk size: {self.chunk_size} words")
            print(f"🧠 Semantic threshold: {self.semantic_chunker.breakpoint_threshold_type} ({self.semantic_chunker.breakpoint_threshold_amount})")
        print("-" * 60)
        
        all_json_files = []
        for input_dir in self.input_dirs:
            if input_dir.exists():
                json_files = list(input_dir.glob("*.json"))
                all_json_files.extend(json_files)
                print(f"📁 Found {len(json_files)} files in {input_dir}")
            else:
                print(f"⚠️  Directory not found: {input_dir}")
        
        if not all_json_files:
            print("❌ No JSON files found in input directories!")
            return
        
        print(f"📄 Total articles to process: {len(all_json_files)}\n")
        
        all_chunks = []
        
        for i, file_path in enumerate(all_json_files, 1):
            print(f"[{i}/{len(all_json_files)}] Processing: {file_path.name}")
            
            chunks = self.process_article(file_path)
            if chunks:
                all_chunks.extend(chunks)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"📊 Progress: {i}/{len(all_json_files)} files processed, {len(all_chunks)} chunks created so far\n")
        
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
            print(f"📏 Words: {chunk['word_count']} (+ {chunk.get('preceding_context_words', 0)} preceding + {chunk.get('following_context_words', 0)} following context)")
            print(f"📍 Position: {chunk.get('chunk_position', 'unknown')}")
            print(f"🔗 URL: {chunk['source_url']}")
            print(f"📅 Date: {chunk['source_date']}")
            print(f"📖 Content preview:")
            print(f"   {chunk['text'][:200]}...")
            if chunk.get('preceding_context'):
                print(f"⬅️  Preceding context: {chunk['preceding_context'][:100]}...")
            if chunk.get('following_context'):
                print(f"➡️  Following context: {chunk['following_context'][:100]}...")
            print("-" * 40)


def main():
    """Main function to run the chunking process."""
    print("Slovak Blog Content Chunker")
    print("=" * 40)
    
    # Initialize chunker with enhanced settings for Slovak content
    chunker = ContentChunker(
        input_dirs=["./data/raw/scraped_data/articles", "./data/raw/scraped_data/pdfs"],
        output_dir="./data/processed/chunked_data",
        chunk_size=800,     # Good balance for Slovak text
        chunk_overlap=200,  # Larger overlap for better context preservation
        context_window=300  # Rich contextual information around each chunk
    )
    
    # Process all articles
    chunker.process_all_articles()
    
    # Save results
    chunker.save_chunks()
    
    # Show preview
    chunker.preview_chunks(3)


if __name__ == "__main__":
    main()