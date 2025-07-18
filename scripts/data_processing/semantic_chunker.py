#!/usr/bin/env python3

import json
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Optional NLTK import - will use fallback if not available
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    
    # Try to download NLTK data if needed (with error handling)
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        try:
            nltk.download('punkt')
        except:
            NLTK_AVAILABLE = False
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except:
        try:
            nltk.download('punkt_tab')
        except:
            NLTK_AVAILABLE = False
            
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available, using regex-based sentence splitting")

class SemanticChunker:
    """
    Semantic chunking implementation that splits text based on semantic similarity
    between sentences, creating chunks at natural semantic boundaries.
    """
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 breakpoint_threshold_type: Literal["percentile", "standard_deviation", "gradient"] = "percentile",
                 breakpoint_threshold_amount: Optional[float] = None,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1500,
                 buffer_size: int = 1):
        """
        Initialize the semantic chunker.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            breakpoint_threshold_type: Method to determine semantic breakpoints
            breakpoint_threshold_amount: Threshold value (default: 95th percentile)
            min_chunk_size: Minimum chunk size in words
            max_chunk_size: Maximum chunk size in words
            buffer_size: Number of sentences to compare on each side (1 = adjacent sentences)
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount or 95.0
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with fallback."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        if NLTK_AVAILABLE:
            try:
                # Use NLTK's sentence tokenizer
                sentences = sent_tokenize(text)
            except Exception as e:
                # Fallback to simple regex-based sentence splitting
                print(f"⚠️  NLTK tokenizer error ({e}), using fallback sentence splitter")
                sentences = self._simple_sentence_split(text)
        else:
            # Use fallback sentence splitter
            sentences = self._simple_sentence_split(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple regex-based sentence splitter as fallback."""
        # Split on sentence endings followed by whitespace and capital letter or number
        # Handle Slovak diacritics and common sentence patterns
        patterns = [
            r'[.!?]+\s+(?=[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ0-9])',  # Standard sentence endings
            r'[.!?]+\s*\n\s*(?=[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ0-9])',  # Sentence endings with newlines
            r'\.\s*\n\s*(?=[A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ0-9])',  # Period followed by newline
        ]
        
        sentences = []
        current_text = text
        
        for pattern in patterns:
            temp_sentences = re.split(pattern, current_text)
            if len(temp_sentences) > 1:
                sentences = temp_sentences
                break
        
        # If no pattern worked, try simple period splitting
        if not sentences or len(sentences) == 1:
            sentences = text.split('. ')
            # Add periods back except for last sentence
            for i in range(len(sentences) - 1):
                sentences[i] += '.'
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _calculate_sentence_distances(self, sentences: List[str]) -> List[float]:
        """
        Calculate semantic distances between consecutive sentence groups.
        
        Returns:
            List of distances between sentence groups
        """
        if len(sentences) <= 2 * self.buffer_size:
            return []
        
        # Encode all sentences
        embeddings = self.embedding_model.encode(sentences)
        
        distances = []
        for i in range(self.buffer_size, len(sentences) - self.buffer_size):
            # Get sentence groups before and after position i
            group_1_start = max(0, i - self.buffer_size)
            group_1_end = i
            group_2_start = i
            group_2_end = min(len(sentences), i + self.buffer_size)
            
            # Calculate mean embeddings for each group
            group_1_embeddings = embeddings[group_1_start:group_1_end]
            group_2_embeddings = embeddings[group_2_start:group_2_end]
            
            group_1_mean = np.mean(group_1_embeddings, axis=0)
            group_2_mean = np.mean(group_2_embeddings, axis=0)
            
            # Calculate cosine distance (1 - similarity)
            similarity = cosine_similarity([group_1_mean], [group_2_mean])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        
        return distances
    
    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        """
        Calculate the threshold for determining breakpoints based on the selected method.
        """
        if not distances:
            return 0.0
        
        if self.breakpoint_threshold_type == "percentile":
            return np.percentile(distances, self.breakpoint_threshold_amount)
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            return mean_distance + (self.breakpoint_threshold_amount * std_distance)
        elif self.breakpoint_threshold_type == "gradient":
            # Use gradient to find sudden changes in distance
            gradients = np.gradient(distances)
            return np.percentile(np.abs(gradients), self.breakpoint_threshold_amount)
        else:
            raise ValueError(f"Unknown threshold type: {self.breakpoint_threshold_type}")
    
    def _merge_sentences_into_chunks(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """
        Merge sentences into chunks based on breakpoints, respecting min/max size constraints.
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Add start and end boundaries
        boundaries = [0] + breakpoints + [len(sentences)]
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Get sentences for this semantic group
            group_sentences = sentences[start_idx:end_idx]
            group_text = ' '.join(group_sentences)
            group_size = len(group_text.split())
            
            # Check if adding this group would exceed max size
            if current_size + group_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= self.min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = group_sentences
                current_size = group_size
            else:
                # Add to current chunk
                current_chunk.extend(group_sentences)
                current_size += group_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """
        Create semantic chunks from text.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks split at semantic boundaries
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 2:
            return [text]
        
        # Calculate distances between sentence groups
        distances = self._calculate_sentence_distances(sentences)
        
        if not distances:
            return [text]
        
        # Calculate threshold for breakpoints
        threshold = self._calculate_breakpoint_threshold(distances)
        
        # Find breakpoints where distance exceeds threshold
        breakpoints = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Adjust index to account for buffer
                breakpoint_idx = i + self.buffer_size
                breakpoints.append(breakpoint_idx)
        
        # Merge sentences into chunks
        chunks = self._merge_sentences_into_chunks(sentences, breakpoints)
        
        return chunks
    
    def chunk_with_metadata(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create semantic chunks with full metadata (compatible with existing system).
        
        Args:
            text: The text to chunk
            metadata: Article metadata (title, url, date, etc.)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        semantic_chunks = self.create_semantic_chunks(text)
        
        chunks_with_metadata = []
        for i, chunk_text in enumerate(semantic_chunks):
            chunk_data = {
                'text': chunk_text,
                'word_count': len(chunk_text.split()),
                'chunk_id': i,
                
                # Source attribution
                'source_url': metadata.get('url', ''),
                'source_title': metadata.get('title', ''),
                'source_date': metadata.get('date', ''),
                'source_scraped_at': metadata.get('scraped_at', ''),
                'source_filename': metadata.get('filename', ''),
                'article_excerpt': metadata.get('excerpt', ''),
                'article_word_count': metadata.get('word_count', 0),
                
                # Chunk positioning
                'chunk_position': 'beginning' if i == 0 else ('end' if i == len(semantic_chunks) - 1 else 'middle'),
                'total_chunks_in_article': len(semantic_chunks),
                
                # Semantic chunking specific metadata
                'chunking_method': 'semantic',
                'chunking_params': {
                    'model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'threshold_type': self.breakpoint_threshold_type,
                    'threshold_amount': self.breakpoint_threshold_amount,
                    'buffer_size': self.buffer_size
                },
                
                # For compatibility with existing system
                'adjacent_chunk_ids': {
                    'previous': i - 1 if i > 0 else None,
                    'next': i + 1 if i < len(semantic_chunks) - 1 else None
                }
            }
            chunks_with_metadata.append(chunk_data)
        
        return chunks_with_metadata


def main():
    """Test the semantic chunker."""
    print("Semantic Text Chunker Test")
    print("=" * 40)
    
    # Sample Slovak health text
    sample_text = """
    Mitochondrie sú často nazývané elektrárňami buniek, pretože produkujú väčšinu energie potrebnej pre bunkové procesy. 
    Tieto organely majú vlastnú DNA a môžu sa samostatne množiť v rámci bunky. Mitochondriálna dysfunkcia je spojená 
    s mnohými ochoreniami vrátane Parkinsonovej choroby a diabetu.
    
    Kvantová biológia je relatívne nová oblasť vedy, ktorá skúma kvantové javy v biologických systémoch. Vedci 
    zistili, že kvantové efekty môžu hrať úlohu vo fotosyntéze, čuchu a dokonca aj v navigácii vtákov. Tieto 
    objavy menia naše chápanie toho, ako funguje život na najzákladnejšej úrovni.
    
    Studená termogenéza je proces, pri ktorom telo produkuje teplo v reakcii na chlad bez triašky. Tento proces 
    primárne prebieha v hnedom tukovom tkanive a môže významne zvýšiť metabolizmus. Pravidelná expozícia chladu 
    môže zlepšiť metabolickú flexibilitu a podporiť zdravie mitochondrií.
    """
    
    # Initialize semantic chunker
    chunker = SemanticChunker(
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85.0,
        min_chunk_size=50,
        max_chunk_size=500
    )
    
    # Create chunks
    chunks = chunker.create_semantic_chunks(sample_text)
    
    print(f"Created {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chunk.split())} words):")
        print(f"{chunk}\n")
        print("-" * 40)


if __name__ == "__main__":
    main()