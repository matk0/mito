#!/usr/bin/env python3
"""
Content Similarity Analyzer
Compares original scraped content with freshly fetched content to identify quality issues.
"""

import re
import math
from typing import Dict, List, Tuple, Set
from collections import Counter
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class ContentSimilarityAnalyzer:
    """Analyzes similarity between original and freshly fetched content"""
    
    def __init__(self):
        self.stop_words_slovak = {
            'a', 'aby', 'aj', 'ako', 'ale', 'alebo', 'and', 'ani', 'áno', 'asi',
            'až', 'bez', 'bude', 'budem', 'budeš', 'budeme', 'budete', 'budú',
            'by', 'bol', 'bola', 'bolo', 'boli', 'byť', 'cez', 'čo', 'či',
            'ďalší', 'ďalšia', 'ďalšie', 'dnes', 'do', 'ho', 'ich', 'ja',
            'je', 'jeho', 'jej', 'jej', 'jeho', 'jem', 'si', 'som', 'sú',
            'ta', 'tak', 'tam', 'te', 'tej', 'ten', 'tento', 'tieto', 'to',
            'toto', 'tu', 'ty', 'v', 've', 'vo', 'za', 'že', 'zo', 'sa',
            'na', 'pre', 'pri', 'od', 'po', 'ku', 'k', 'o', 's', 'so', 'no'
        }
        
    def compare_articles(self, original: Dict, fetched: Dict) -> Dict:
        """Compare original and fetched article content"""
        result = {
            'similarity_score': 0.0,
            'length_ratio': 0.0,
            'word_count_ratio': 0.0,
            'content_overlap': 0.0,
            'title_similarity': 0.0,
            'issues_detected': [],
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        original_content = original.get('content', '') or ''
        fetched_content = fetched.get('content', '') or ''
        
        original_title = original.get('title', '') or ''
        fetched_title = fetched.get('title', '') or ''
        
        # Basic length comparison
        orig_len = len(original_content)
        fetch_len = len(fetched_content)
        
        if orig_len > 0 and fetch_len > 0:
            result['length_ratio'] = min(orig_len, fetch_len) / max(orig_len, fetch_len)
        elif orig_len == 0 and fetch_len == 0:
            result['length_ratio'] = 1.0
        else:
            result['length_ratio'] = 0.0
            
        # Word count comparison
        orig_words = len(original_content.split())
        fetch_words = len(fetched_content.split())
        
        if orig_words > 0 and fetch_words > 0:
            result['word_count_ratio'] = min(orig_words, fetch_words) / max(orig_words, fetch_words)
        elif orig_words == 0 and fetch_words == 0:
            result['word_count_ratio'] = 1.0
        else:
            result['word_count_ratio'] = 0.0
            
        # Content similarity analysis
        if original_content and fetched_content:
            result['content_overlap'] = self._calculate_content_overlap(original_content, fetched_content)
            result['similarity_score'] = self._calculate_semantic_similarity(original_content, fetched_content)
        
        # Title similarity
        if original_title and fetched_title:
            result['title_similarity'] = self._calculate_text_similarity(original_title, fetched_title)
        
        # Detailed analysis
        result['detailed_analysis'] = self._detailed_content_analysis(original_content, fetched_content)
        
        # Issue detection
        result['issues_detected'] = self._detect_content_issues(original, fetched, result)
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
        
    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """Calculate content overlap using set intersection"""
        if not text1 or not text2:
            return 0.0
            
        # Normalize and tokenize
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())
        
        # Remove stop words
        words1 = words1 - self.stop_words_slovak
        words2 = words2 - self.stop_words_slovak
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using cosine similarity of word vectors"""
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)
        
        # Create word frequency vectors
        words1 = text1_norm.split()
        words2 = text2_norm.split()
        
        # Remove stop words
        words1 = [w for w in words1 if w not in self.stop_words_slovak]
        words2 = [w for w in words2 if w not in self.stop_words_slovak]
        
        if not words1 or not words2:
            return 0.0
            
        # Create vocabulary
        vocab = set(words1 + words2)
        
        # Create frequency vectors
        freq1 = Counter(words1)
        freq2 = Counter(words2)
        
        vector1 = [freq1.get(word, 0) for word in vocab]
        vector2 = [freq2.get(word, 0) for word in vocab]
        
        # Calculate cosine similarity
        return self._cosine_similarity(vector1, vector2)
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching"""
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts
        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)
        
        # Use SequenceMatcher
        matcher = SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep Slovak characters
        text = re.sub(r'[^\w\sáäčďéěíĺľňóôŕšťúůýž]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vector1) != len(vector2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
        
    def _detailed_content_analysis(self, original: str, fetched: str) -> Dict:
        """Perform detailed content analysis"""
        analysis = {
            'original_stats': self._get_text_stats(original),
            'fetched_stats': self._get_text_stats(fetched),
            'content_structure_comparison': {},
            'unique_content': {}
        }
        
        if original and fetched:
            # Analyze content structure
            analysis['content_structure_comparison'] = self._compare_content_structure(original, fetched)
            
            # Find unique content in each version
            analysis['unique_content'] = self._find_unique_content(original, fetched)
            
        return analysis
        
    def _get_text_stats(self, text: str) -> Dict:
        """Get basic text statistics"""
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_word_length': 0.0,
                'avg_sentence_length': 0.0
            }
            
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0.0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0.0
        }
        
    def _compare_content_structure(self, original: str, fetched: str) -> Dict:
        """Compare content structure between versions"""
        orig_stats = self._get_text_stats(original)
        fetch_stats = self._get_text_stats(fetched)
        
        comparison = {}
        
        for key in orig_stats:
            orig_val = orig_stats[key]
            fetch_val = fetch_stats[key]
            
            if orig_val > 0 and fetch_val > 0:
                ratio = min(orig_val, fetch_val) / max(orig_val, fetch_val)
            elif orig_val == 0 and fetch_val == 0:
                ratio = 1.0
            else:
                ratio = 0.0
                
            comparison[key + '_ratio'] = ratio
            comparison[key + '_difference'] = abs(orig_val - fetch_val)
            
        return comparison
        
    def _find_unique_content(self, original: str, fetched: str) -> Dict:
        """Find content unique to each version"""
        if not original or not fetched:
            return {'original_unique': [], 'fetched_unique': []}
            
        # Split into sentences
        orig_sentences = [s.strip() for s in re.split(r'[.!?]+', original) if s.strip()]
        fetch_sentences = [s.strip() for s in re.split(r'[.!?]+', fetched) if s.strip()]
        
        # Normalize sentences for comparison
        orig_norm = [self._normalize_text(s) for s in orig_sentences]
        fetch_norm = [self._normalize_text(s) for s in fetch_sentences]
        
        # Find unique sentences (using fuzzy matching)
        orig_unique = []
        fetch_unique = []
        
        for i, orig_sent in enumerate(orig_norm):
            if not any(self._calculate_text_similarity(orig_sent, fetch_sent) > 0.8 
                      for fetch_sent in fetch_norm):
                orig_unique.append(orig_sentences[i])
                
        for i, fetch_sent in enumerate(fetch_norm):
            if not any(self._calculate_text_similarity(fetch_sent, orig_sent) > 0.8 
                      for orig_sent in orig_norm):
                fetch_unique.append(fetch_sentences[i])
                
        return {
            'original_unique': orig_unique[:5],  # Limit to first 5
            'fetched_unique': fetch_unique[:5]
        }
        
    def _detect_content_issues(self, original: Dict, fetched: Dict, comparison: Dict) -> List[Dict]:
        """Detect specific content issues"""
        issues = []
        
        # Length ratio issues
        length_ratio = comparison.get('length_ratio', 0.0)
        if length_ratio < 0.5:
            severity = 'critical' if length_ratio < 0.2 else 'major'
            issues.append({
                'type': 'length_mismatch',
                'severity': severity,
                'description': f'Significant length difference (ratio: {length_ratio:.2f})',
                'confidence': 0.9
            })
            
        # Content overlap issues
        overlap = comparison.get('content_overlap', 0.0)
        if overlap < 0.3:
            severity = 'critical' if overlap < 0.1 else 'major'
            issues.append({
                'type': 'content_mismatch',
                'severity': severity,
                'description': f'Low content overlap (overlap: {overlap:.2f})',
                'confidence': 0.8
            })
            
        # Title mismatch
        title_sim = comparison.get('title_similarity', 0.0)
        if title_sim < 0.7:
            issues.append({
                'type': 'title_mismatch',
                'severity': 'minor',
                'description': f'Title differs between versions (similarity: {title_sim:.2f})',
                'confidence': 0.7
            })
            
        # Paywall detection differences
        orig_paywall = self._simple_paywall_check(original.get('content', ''))
        fetch_paywall = fetched.get('paywall_detected', False)
        
        if orig_paywall != fetch_paywall:
            issues.append({
                'type': 'paywall_status_change',
                'severity': 'major',
                'description': f'Paywall status changed: original={orig_paywall}, fetched={fetch_paywall}',
                'confidence': 0.8
            })
            
        # Word count dramatic change
        word_ratio = comparison.get('word_count_ratio', 0.0)
        if word_ratio < 0.3:
            issues.append({
                'type': 'word_count_change',
                'severity': 'major',
                'description': f'Dramatic word count change (ratio: {word_ratio:.2f})',
                'confidence': 0.9
            })
            
        return issues
        
    def _simple_paywall_check(self, content: str) -> bool:
        """Simple paywall detection for comparison"""
        if not content:
            return False
            
        paywall_indicators = [
            'prémium členov', 'premium členov', 'platených členov',
            'registrácia', 'prihlásiť sa', 'predplatné',
            'webinár', 'pokračovanie článku'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in paywall_indicators)
        
    def _generate_recommendations(self, comparison: Dict) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        # Low similarity recommendations
        similarity = comparison.get('similarity_score', 0.0)
        if similarity < 0.5:
            recommendations.append("Consider re-scraping this article - content differs significantly")
            
        # Length mismatch recommendations
        length_ratio = comparison.get('length_ratio', 0.0)
        if length_ratio < 0.7:
            recommendations.append("Investigate potential content truncation or extraction issues")
            
        # Content overlap recommendations
        overlap = comparison.get('content_overlap', 0.0)
        if overlap < 0.4:
            recommendations.append("Check for JavaScript-rendered content or dynamic loading")
            
        # Critical issues recommendations
        critical_issues = [issue for issue in comparison.get('issues_detected', []) 
                          if issue.get('severity') == 'critical']
        if critical_issues:
            recommendations.append("Critical content issues detected - manual review recommended")
            
        # No issues found
        if not comparison.get('issues_detected') and similarity > 0.8:
            recommendations.append("Content quality appears good - suitable for GraphRAG")
            
        return recommendations
        
    def batch_compare(self, articles_comparison: List[Tuple[Dict, Dict]]) -> List[Dict]:
        """Compare multiple articles in batch"""
        results = []
        
        for i, (original, fetched) in enumerate(articles_comparison):
            logger.debug(f"Comparing article {i+1}/{len(articles_comparison)}")
            comparison = self.compare_articles(original, fetched)
            comparison['article_index'] = i
            results.append(comparison)
            
        return results