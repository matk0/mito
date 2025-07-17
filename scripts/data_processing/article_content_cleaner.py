#!/usr/bin/env python3
"""
Article Content Cleaner
Analyzes scraped articles to identify and remove noise, repetitive marketing terms,
and non-core content to prepare high-quality knowledge base content.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleContentCleaner:
    """Analyzes and cleans article content to remove noise and marketing terms"""
    
    def __init__(self, articles_dir: str, output_dir: str = None):
        self.articles_dir = Path(articles_dir)
        self.output_dir = Path(output_dir) if output_dir else self.articles_dir / "cleaned_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis results
        self.word_frequency = Counter()
        self.phrase_frequency = Counter()
        self.marketing_patterns = Counter()
        self.noise_patterns = Counter()
        self.article_analyses = []
        
        # Define noise and marketing patterns in Slovak/Czech
        self.marketing_keywords = {
            # Direct marketing
            'registrácia', 'registracia', 'prihlásiť', 'prihlásenie', 'prihlasenie',
            'predplatné', 'predplatne', 'webinár', 'webinar', 'členstvo', 'členstvo',
            'premium', 'prémium', 'zlatý', 'zlaty', 'vip', 'exclusive', 'exkluzívny',
            
            # Call to action
            'klikni', 'stiahnuť', 'stiahnúť', 'download', 'pridaj sa', 'pripoj sa',
            'registruj', 'objednaj', 'kúp', 'kup', 'získaj', 'ziskaj',
            
            # Blog/social marketing
            'zdieľaj', 'zdielaj', 'share', 'like', 'sleduj', 'follow', 'subscribe',
            'odber', 'newsletter', 'notifikácie', 'notifikacie',
            
            # Promotional
            'zdarma', 'zadarmo', 'free', 'akcia', 'zľava', 'zlava', 'discount',
            'limited', 'limitovaný', 'limitovany', 'ponuka', 'offer'
        }
        
        self.reference_patterns = [
            r'\[R\]', r'\[r\]', r'\[\d+\]', r'http[s]?://[^\s]+',
            r'www\.[^\s]+', r'\bhttps://[^\s]+\b'
        ]
        
        self.navigation_patterns = {
            'pokračovanie', 'pokracovanie', 'continued', 'next', 'previous', 'ďalej', 'dalej',
            'späť', 'spat', 'back', 'home', 'domov', 'menu', 'navigácia', 'navigacia'
        }
        
        self.meta_patterns = {
            'autor', 'author', 'dátum', 'datum', 'date', 'publikované', 'publikovane',
            'published', 'kategória', 'kategoria', 'category', 'tag', 'tagy',
            'komentáre', 'komentare', 'comments', 'views', 'zobrazenia'
        }
        
    def load_articles(self) -> List[Dict]:
        """Load all article JSON files"""
        logger.info(f"Loading articles from {self.articles_dir}")
        articles = []
        json_files = list(self.articles_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    article['source_file'] = file_path.name
                    articles.append(article)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        logger.info(f"Loaded {len(articles)} articles")
        return articles
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Remove multiple whitespace/newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs and references
        for pattern in self.reference_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_phrases(self, text: str, min_length: int = 3, max_length: int = 8) -> List[str]:
        """Extract meaningful phrases from text"""
        if not text:
            return []
        
        # Split into sentences and clean
        sentences = re.split(r'[.!?]+', text.lower())
        phrases = []
        
        for sentence in sentences:
            words = re.findall(r'\b\w{3,}\b', sentence)  # Only words 3+ chars
            
            # Extract n-grams
            for length in range(min_length, min(max_length + 1, len(words) + 1)):
                for i in range(len(words) - length + 1):
                    phrase = ' '.join(words[i:i + length])
                    if self.is_meaningful_phrase(phrase):
                        phrases.append(phrase)
        
        return phrases
    
    def is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if phrase is meaningful (not just function words)"""
        # Function words in Slovak/Czech that shouldn't form meaningful phrases alone
        function_words = {
            'a', 'ale', 'alebo', 'ako', 'ak', 'aby', 'až', 'ani', 'už', 'už',
            'v', 'vo', 'na', 'z', 'za', 'do', 'od', 'po', 'pre', 'pri', 'cez',
            'to', 'ta', 'tie', 'tá', 'ten', 'toto', 'táto', 'tento',
            'je', 'sú', 'som', 'si', 'sa', 'sme', 'ste', 'má', 'mať', 'môže',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would'
        }
        
        words = phrase.split()
        
        # Must have at least one non-function word
        meaningful_words = [w for w in words if w not in function_words and len(w) > 3]
        
        return len(meaningful_words) >= 1 and len(words) >= 2
    
    def identify_marketing_content(self, text: str) -> Dict:
        """Identify marketing and promotional content"""
        if not text:
            return {'score': 0, 'patterns': [], 'confidence': 0.0}
        
        text_lower = text.lower()
        found_patterns = []
        score = 0
        
        # Check for marketing keywords
        for keyword in self.marketing_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                found_patterns.append(f"Marketing keyword '{keyword}': {count} times")
                score += count * 2  # Weight marketing terms higher
        
        # Check for navigation patterns
        for pattern in self.navigation_patterns:
            count = text_lower.count(pattern)
            if count > 0:
                found_patterns.append(f"Navigation pattern '{pattern}': {count} times")
                score += count
        
        # Check for meta patterns
        for pattern in self.meta_patterns:
            count = text_lower.count(pattern)
            if count > 0:
                found_patterns.append(f"Meta pattern '{pattern}': {count} times")
                score += count
        
        # Check for repetitive self-references
        self_ref_patterns = [
            r'\bblog\w*\b', r'\bčlán\w*\b', r'\barticle\w*\b',
            r'\bweb\w*\b', r'\bstránk\w*\b', r'\bsite\w*\b'
        ]
        
        for pattern in self_ref_patterns:
            matches = re.findall(pattern, text_lower)
            if len(matches) > 3:  # If mentioned more than 3 times
                found_patterns.append(f"Self-reference pattern '{pattern}': {len(matches)} times")
                score += len(matches)
        
        # Normalize score by text length
        words_count = len(text.split())
        confidence = min(score / max(words_count / 100, 1), 1.0)  # Normalize to 0-1
        
        return {
            'score': score,
            'patterns': found_patterns,
            'confidence': confidence
        }
    
    def analyze_content_quality(self, content: str) -> Dict:
        """Analyze the overall content quality"""
        if not content:
            return {'quality_score': 0.0, 'issues': ['Empty content']}
        
        issues = []
        quality_score = 1.0
        
        # Basic metrics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Check content structure
        if word_count < 100:
            issues.append('Very short content')
            quality_score -= 0.3
        elif word_count < 300:
            issues.append('Short content')
            quality_score -= 0.1
        
        if sentence_count < 5:
            issues.append('Very few sentences')
            quality_score -= 0.2
        
        if paragraph_count < 2:
            issues.append('Poor paragraph structure')
            quality_score -= 0.1
        
        # Check for repetitive content
        words = content.lower().split()
        unique_words = set(words)
        
        if len(unique_words) < len(words) * 0.4:  # Less than 40% unique words
            issues.append('High word repetition')
            quality_score -= 0.2
        
        # Check sentence length variation
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            if avg_length < 5:
                issues.append('Very short sentences')
                quality_score -= 0.1
            elif avg_length > 30:
                issues.append('Very long sentences')
                quality_score -= 0.1
        
        return {
            'quality_score': max(0.0, quality_score),
            'issues': issues,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'unique_word_ratio': len(unique_words) / len(words) if words else 0
        }
    
    def analyze_article(self, article: Dict) -> Dict:
        """Perform comprehensive analysis on a single article"""
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Clean content
        cleaned_content = self.clean_text(content)
        
        # Analyze marketing content
        marketing_analysis = self.identify_marketing_content(cleaned_content)
        
        # Analyze content quality
        quality_analysis = self.analyze_content_quality(cleaned_content)
        
        # Extract and count phrases
        phrases = self.extract_phrases(cleaned_content)
        
        # Update global counters
        words = re.findall(r'\b\w{4,}\b', cleaned_content.lower())  # Words 4+ chars
        self.word_frequency.update(words)
        self.phrase_frequency.update(phrases)
        
        # Determine if article needs cleaning
        needs_cleaning = (
            marketing_analysis['confidence'] > 0.3 or
            quality_analysis['quality_score'] < 0.6 or
            len(quality_analysis['issues']) > 2
        )
        
        # Calculate overall cleanliness score
        cleanliness_score = (
            (1.0 - marketing_analysis['confidence']) * 0.4 +
            quality_analysis['quality_score'] * 0.6
        )
        
        result = {
            'url': article.get('url', 'unknown'),
            'title': title,
            'source_file': article.get('source_file', 'unknown'),
            'original_word_count': article.get('word_count', 0),
            'cleaned_word_count': len(cleaned_content.split()),
            'marketing_analysis': marketing_analysis,
            'quality_analysis': quality_analysis,
            'needs_cleaning': needs_cleaning,
            'cleanliness_score': cleanliness_score,
            'top_phrases': phrases[:10],  # Top 10 phrases for this article
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_analysis(self) -> Dict:
        """Run complete content analysis"""
        logger.info("Starting content analysis...")
        start_time = datetime.now()
        
        articles = self.load_articles()
        
        # Analyze each article
        for i, article in enumerate(articles):
            logger.info(f"Analyzing article {i+1}/{len(articles)}: {article.get('title', 'unknown')[:50]}...")
            result = self.analyze_article(article)
            self.article_analyses.append(result)
        
        end_time = datetime.now()
        
        # Generate comprehensive summary
        summary = self._generate_summary(start_time, end_time)
        
        # Save results
        self._save_results()
        
        logger.info(f"Content analysis complete. Analyzed {len(articles)} articles")
        return summary
    
    def _generate_summary(self, start_time: datetime, end_time: datetime) -> Dict:
        """Generate comprehensive summary"""
        total_articles = len(self.article_analyses)
        
        if total_articles == 0:
            return {'error': 'No articles analyzed'}
        
        # Calculate statistics
        needs_cleaning_count = sum(1 for a in self.article_analyses if a['needs_cleaning'])
        avg_cleanliness = sum(a['cleanliness_score'] for a in self.article_analyses) / total_articles
        
        # Quality distribution
        quality_ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        quality_distribution = {}
        for low, high in quality_ranges:
            count = sum(1 for a in self.article_analyses 
                       if low <= a['quality_analysis']['quality_score'] < high)
            quality_distribution[f'{low:.1f}-{high:.1f}'] = count
        
        # Marketing content distribution
        high_marketing = sum(1 for a in self.article_analyses 
                           if a['marketing_analysis']['confidence'] > 0.5)
        medium_marketing = sum(1 for a in self.article_analyses 
                             if 0.2 < a['marketing_analysis']['confidence'] <= 0.5)
        low_marketing = total_articles - high_marketing - medium_marketing
        
        # Find most common noise patterns
        all_marketing_patterns = []
        for analysis in self.article_analyses:
            all_marketing_patterns.extend(analysis['marketing_analysis']['patterns'])
        
        marketing_pattern_freq = Counter(all_marketing_patterns)
        
        summary = {
            'analysis_info': {
                'total_articles': total_articles,
                'analysis_start': start_time.isoformat(),
                'analysis_end': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds()
            },
            'cleaning_recommendations': {
                'articles_needing_cleaning': needs_cleaning_count,
                'cleaning_percentage': (needs_cleaning_count / total_articles) * 100,
                'average_cleanliness_score': avg_cleanliness
            },
            'quality_distribution': quality_distribution,
            'marketing_content_distribution': {
                'high_marketing': high_marketing,
                'medium_marketing': medium_marketing,
                'low_marketing': low_marketing
            },
            'most_common_noise_patterns': dict(marketing_pattern_freq.most_common(20)),
            'most_frequent_words': dict(self.word_frequency.most_common(50)),
            'most_frequent_phrases': dict(self.phrase_frequency.most_common(30)),
            'articles_by_cleanliness': {
                'cleanest': sorted(self.article_analyses, 
                                 key=lambda x: x['cleanliness_score'], reverse=True)[:10],
                'needs_most_cleaning': sorted(self.article_analyses, 
                                            key=lambda x: x['cleanliness_score'])[:10]
            }
        }
        
        return summary
    
    def _save_results(self):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed analysis
        detailed_file = self.output_dir / f"content_analysis_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'article_analyses': self.article_analyses,
                'word_frequency': dict(self.word_frequency.most_common(200)),
                'phrase_frequency': dict(self.phrase_frequency.most_common(100))
            }, f, indent=2, ensure_ascii=False)
        
        # Save clean articles list (for knowledge base)
        clean_articles = [a for a in self.article_analyses if a['cleanliness_score'] > 0.7]
        clean_file = self.output_dir / f"clean_articles_{timestamp}.json"
        with open(clean_file, 'w', encoding='utf-8') as f:
            json.dump(clean_articles, f, indent=2, ensure_ascii=False)
        
        # Save articles needing cleaning
        dirty_articles = [a for a in self.article_analyses if a['needs_cleaning']]
        dirty_file = self.output_dir / f"articles_need_cleaning_{timestamp}.json"
        with open(dirty_file, 'w', encoding='utf-8') as f:
            json.dump(dirty_articles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Clean articles: {len(clean_articles)}")
        logger.info(f"Articles needing cleaning: {len(dirty_articles)}")

def main():
    parser = argparse.ArgumentParser(description='Article Content Cleaner and Analyzer')
    parser.add_argument('--articles-dir', '-d', 
                       default='./data/raw/scraped_data/articles',
                       help='Directory containing article JSON files')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for analysis reports')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = ArticleContentCleaner(args.articles_dir, args.output_dir)
    
    # Run analysis
    summary = cleaner.run_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("ARTICLE CONTENT ANALYSIS SUMMARY")
    print("="*80)
    
    if 'error' in summary:
        print(f"Error: {summary['error']}")
        return
    
    info = summary['analysis_info']
    cleaning = summary['cleaning_recommendations']
    
    print(f"Total articles analyzed: {info['total_articles']}")
    print(f"Analysis duration: {info['duration_seconds']:.1f} seconds")
    print()
    
    print("CLEANING RECOMMENDATIONS:")
    print(f"Articles needing cleaning: {cleaning['articles_needing_cleaning']} ({cleaning['cleaning_percentage']:.1f}%)")
    print(f"Average cleanliness score: {cleaning['average_cleanliness_score']:.3f}")
    print()
    
    print("QUALITY DISTRIBUTION:")
    for range_name, count in summary['quality_distribution'].items():
        percentage = (count / info['total_articles']) * 100
        print(f"  {range_name}: {count} articles ({percentage:.1f}%)")
    print()
    
    print("MARKETING CONTENT DISTRIBUTION:")
    marketing = summary['marketing_content_distribution']
    total = info['total_articles']
    print(f"  High marketing content: {marketing['high_marketing']} ({marketing['high_marketing']/total*100:.1f}%)")
    print(f"  Medium marketing content: {marketing['medium_marketing']} ({marketing['medium_marketing']/total*100:.1f}%)")
    print(f"  Low marketing content: {marketing['low_marketing']} ({marketing['low_marketing']/total*100:.1f}%)")
    print()
    
    print("TOP NOISE PATTERNS:")
    for pattern, count in list(summary['most_common_noise_patterns'].items())[:10]:
        print(f"  {pattern}: {count} times")
    print()
    
    print("CLEANEST ARTICLES:")
    for article in summary['articles_by_cleanliness']['cleanest'][:5]:
        print(f"  {article['cleanliness_score']:.3f} - {article['title'][:60]}...")
    print()
    
    print("ARTICLES NEEDING MOST CLEANING:")
    for article in summary['articles_by_cleanliness']['needs_most_cleaning'][:5]:
        print(f"  {article['cleanliness_score']:.3f} - {article['title'][:60]}...")
    
    print(f"\nDetailed results saved to: {cleaner.output_dir}")

if __name__ == "__main__":
    main()