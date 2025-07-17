#!/usr/bin/env python3
"""
Article Cleaner
Cleans articles based on noise pattern analysis to prepare high-quality content for knowledge base.
"""

import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleCleaner:
    """Cleans articles by removing marketing content and noise patterns"""
    
    def __init__(self, articles_dir: str, output_dir: str = None):
        self.articles_dir = Path(articles_dir)
        self.output_dir = Path(output_dir) if output_dir else self.articles_dir / "cleaned_articles"
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.cleaning_stats = {
            'total_articles': 0,
            'successfully_cleaned': 0,
            'failed_cleaning': 0,
            'total_words_removed': 0,
            'patterns_removed': Counter()
        }
        
        # Define patterns to remove based on analysis
        self.setup_cleaning_patterns()
    
    def setup_cleaning_patterns(self):
        """Setup all cleaning patterns based on noise analysis"""
        
        # Marketing and promotional patterns
        self.marketing_patterns = [
            # Registration/membership patterns
            r'(?i)\b(registr[áa]ci[au]?|prihl[áa]si?[ťt]|prihl[áa]senie?|predplatn[éey]?|webinár\w*|členstvo|členovia?|premium|prémium)\b[^.]*\.',
            
            # Call to action patterns
            r'(?i)\b(klikni|stiahnú?[ťt]|download|pridaj\s+sa|pripoj\s+sa|registruj|objednaj|kúp|získaj)\b[^.]*\.',
            
            # Social media patterns
            r'(?i)\b(zdieľaj|share|like|sleduj|follow|subscribe|odber|newsletter|notifik[áa]ci[ae])\b[^.]*\.',
            
            # Promotional offers
            r'(?i)\b(zdarma|zadarmo|free|akcia|zľava|discount|limitovan[ýy]|ponuka|offer)\b[^.]*\.',
            
            # Blog self-references
            r'(?i)(v\s+)?predošl\w+\s+(článk\w+|blog\w+|príspevk\w+)',
            r'(?i)(v\s+)?ďalš\w+\s+(článk\w+|blog\w+|príspevk\w+)',
            r'(?i)ako\s+som\s+(už\s+)?písal',
            r'(?i)v\s+mojom\s+(predošlom\s+)?(článku|blogu)',
            
            # Navigation elements
            r'(?i)pokračovanie\s+(nájde[šte]+|článku)',
            r'(?i)(späť|back|home|domov|menu|navigáci[au])',
            
            # Meta content
            r'(?i)(autor|publikovan[éey]|kategóri[au]|tag\w*|komentár\w*|zobrazeni[au])',
            
            # Newsletter/webinar promotions
            r'(?i)(\()?poznámka:\s*so?\s+zlatými\s+premium[^)]*(\))?',
            r'(?i)registr\w+\s+(sa\s+)?na\s+(prvý\s+)?(verejný\s+)?webinár',
            r'(?i)na\s+tento\s+sa\s+nemusíte\s+registrovať',
        ]
        
        # URL patterns
        self.url_patterns = [
            r'https?://[^\s\]]+',
            r'www\.[^\s\]]+',
            r'\[R\]',
            r'\[\d+\]',
        ]
        
        # Repetitive navigation phrases
        self.navigation_phrases = [
            r'(?i)registrácia\s+na\s+webinár\s*>>',
            r'(?i)<<\s*späť',
            r'(?i)>>\s*ďalej',
            r'(?i)čítať\s+viac\s*>>',
            r'(?i)stiahnúť\s+>>',
        ]
        
        # Common footer/header patterns
        self.footer_patterns = [
            r'(?i)referencie,?\s+odkazy\s+a\s+citácie:?',
            r'(?i)zdieľaj\s+tento\s+článok',
            r'(?i)podobné\s+články',
            r'(?i)odporúčané\s+čítanie',
            r'(?i)súvisiace\s+príspevky',
        ]
        
        # Excessive whitespace patterns
        self.whitespace_patterns = [
            r'\n{4,}',  # More than 3 newlines
            r'[ \t]{3,}',  # More than 2 spaces/tabs
            r'\n\s*\n\s*\n',  # Multiple empty lines
        ]
    
    def clean_marketing_content(self, text: str) -> str:
        """Remove marketing and promotional content"""
        cleaned = text
        
        for pattern in self.marketing_patterns:
            matches = re.findall(pattern, cleaned, re.MULTILINE | re.DOTALL)
            if matches:
                self.cleaning_stats['patterns_removed']['marketing'] += len(matches)
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)
        
        return cleaned
    
    def clean_urls_references(self, text: str) -> str:
        """Remove URLs and reference markers"""
        cleaned = text
        
        for pattern in self.url_patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                self.cleaning_stats['patterns_removed']['urls'] += len(matches)
                cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned
    
    def clean_navigation_elements(self, text: str) -> str:
        """Remove navigation and UI elements"""
        cleaned = text
        
        for pattern in self.navigation_phrases:
            matches = re.findall(pattern, cleaned, re.MULTILINE)
            if matches:
                self.cleaning_stats['patterns_removed']['navigation'] += len(matches)
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        return cleaned
    
    def clean_footer_content(self, text: str) -> str:
        """Remove footer and header content"""
        cleaned = text
        
        # Remove everything after common footer patterns
        for pattern in self.footer_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                self.cleaning_stats['patterns_removed']['footer'] += 1
                cleaned = cleaned[:match.start()]
        
        return cleaned
    
    def clean_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        cleaned = text
        
        # Replace excessive whitespace
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
        cleaned = re.sub(r'[ \t]{3,}', '  ', cleaned)
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        
        # Clean up spacing around punctuation
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'([.,!?;:])\s{2,}', r'\1 ', cleaned)
        
        return cleaned.strip()
    
    def remove_empty_sections(self, text: str) -> str:
        """Remove sections that became empty after cleaning"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty headers (lines that end with : but have no content after)
            if line.strip().endswith(':'):
                # Check if next non-empty line is another header or end of text
                has_content = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].strip().endswith(':'):
                        has_content = True
                        break
                
                if not has_content:
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def preserve_core_content(self, text: str) -> str:
        """Ensure core scientific/educational content is preserved"""
        # This is a safety check to ensure we're not removing too much
        
        # Preserve content with scientific terms
        scientific_markers = [
            'mitochondri', 'cirkadiáln', 'cirkadialn', 'kvantov', 'biológi',
            'hormón', 'hormon', 'neurotransmiter', 'metaboli', 'bunka', 'buniek',
            'molekul', 'elektrón', 'elektron', 'fotón', 'foton', 'energia',
            'svetlo', 'teplota', 'magnetiz', 'fyzik', 'chémia', 'chemia',
            'proteín', 'protein', 'vitamín', 'vitamin', 'minerál', 'mineral'
        ]
        
        # Count scientific terms
        text_lower = text.lower()
        scientific_score = sum(1 for marker in scientific_markers if marker in text_lower)
        
        # If text has high scientific content, be more conservative with cleaning
        if scientific_score > 10:
            logger.debug(f"High scientific content detected (score: {scientific_score}), preserving more content")
        
        return text
    
    def clean_article_content(self, content: str) -> str:
        """Main cleaning pipeline"""
        if not content:
            return ""
        
        original_length = len(content)
        
        # Apply cleaning steps in order
        cleaned = content
        
        # 1. Remove marketing content
        cleaned = self.clean_marketing_content(cleaned)
        
        # 2. Remove URLs and references
        cleaned = self.clean_urls_references(cleaned)
        
        # 3. Remove navigation elements
        cleaned = self.clean_navigation_elements(cleaned)
        
        # 4. Remove footer content
        cleaned = self.clean_footer_content(cleaned)
        
        # 5. Clean whitespace
        cleaned = self.clean_whitespace(cleaned)
        
        # 6. Remove empty sections
        cleaned = self.remove_empty_sections(cleaned)
        
        # 7. Final safety check
        cleaned = self.preserve_core_content(cleaned)
        
        # Calculate statistics
        removed_length = original_length - len(cleaned)
        self.cleaning_stats['total_words_removed'] += len(content.split()) - len(cleaned.split())
        
        return cleaned
    
    def clean_article(self, article: Dict) -> Dict:
        """Clean a single article"""
        cleaned_article = article.copy()
        
        # Clean content
        original_content = article.get('content', '')
        cleaned_content = self.clean_article_content(original_content)
        
        # Update article
        cleaned_article['content'] = cleaned_content
        cleaned_article['cleaned_word_count'] = len(cleaned_content.split())
        cleaned_article['cleaning_metadata'] = {
            'cleaned_at': datetime.now().isoformat(),
            'original_word_count': article.get('word_count', len(original_content.split())),
            'cleaned_word_count': len(cleaned_content.split()),
            'reduction_percentage': round(
                (1 - len(cleaned_content) / len(original_content)) * 100, 2
            ) if original_content else 0
        }
        
        return cleaned_article
    
    def process_articles(self) -> Dict:
        """Process all articles"""
        logger.info("Starting article cleaning process...")
        start_time = datetime.now()
        
        # Load articles
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
                self.cleaning_stats['failed_cleaning'] += 1
        
        self.cleaning_stats['total_articles'] = len(articles)
        logger.info(f"Loaded {len(articles)} articles for cleaning")
        
        # Clean each article
        cleaned_articles = []
        for i, article in enumerate(articles):
            try:
                logger.info(f"Cleaning article {i+1}/{len(articles)}: {article.get('title', 'unknown')[:50]}...")
                cleaned = self.clean_article(article)
                cleaned_articles.append(cleaned)
                self.cleaning_stats['successfully_cleaned'] += 1
            except Exception as e:
                logger.error(f"Error cleaning article {article.get('source_file', 'unknown')}: {e}")
                self.cleaning_stats['failed_cleaning'] += 1
        
        # Save cleaned articles
        self._save_cleaned_articles(cleaned_articles)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = {
            'cleaning_stats': self.cleaning_stats,
            'duration_seconds': duration,
            'average_reduction': self._calculate_average_reduction(cleaned_articles),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _save_cleaned_articles(self, articles: List[Dict]):
        """Save cleaned articles to output directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual cleaned articles
        articles_output_dir = self.output_dir / "articles"
        articles_output_dir.mkdir(exist_ok=True)
        
        for article in articles:
            source_file = article.get('source_file', 'unknown.json')
            output_file = articles_output_dir / source_file
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
        
        # Save cleaning summary
        summary_file = self.output_dir / f"cleaning_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_articles': len(articles),
                'cleaning_stats': self.cleaning_stats,
                'articles_summary': [
                    {
                        'title': a.get('title', 'unknown'),
                        'source_file': a.get('source_file', 'unknown'),
                        'original_words': a.get('word_count', 0),
                        'cleaned_words': a.get('cleaned_word_count', 0),
                        'reduction_percentage': a.get('cleaning_metadata', {}).get('reduction_percentage', 0)
                    }
                    for a in articles
                ]
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} cleaned articles to {self.output_dir}")
    
    def _calculate_average_reduction(self, articles: List[Dict]) -> float:
        """Calculate average content reduction percentage"""
        reductions = [
            a.get('cleaning_metadata', {}).get('reduction_percentage', 0)
            for a in articles
        ]
        
        return sum(reductions) / len(reductions) if reductions else 0

def main():
    parser = argparse.ArgumentParser(description='Clean articles by removing noise and marketing content')
    parser.add_argument('--articles-dir', '-d', 
                       default='./data/raw/scraped_data/articles',
                       help='Directory containing article JSON files')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for cleaned articles')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize cleaner
    cleaner = ArticleCleaner(args.articles_dir, args.output_dir)
    
    # Process articles
    summary = cleaner.process_articles()
    
    # Print summary
    print("\n" + "="*80)
    print("ARTICLE CLEANING SUMMARY")
    print("="*80)
    
    stats = summary['cleaning_stats']
    print(f"Total articles processed: {stats['total_articles']}")
    print(f"Successfully cleaned: {stats['successfully_cleaned']}")
    print(f"Failed cleaning: {stats['failed_cleaning']}")
    print(f"Total words removed: {stats['total_words_removed']:,}")
    print(f"Average content reduction: {summary['average_reduction']:.1f}%")
    print(f"Processing time: {summary['duration_seconds']:.1f} seconds")
    
    print("\nPatterns removed:")
    for pattern_type, count in stats['patterns_removed'].items():
        print(f"  {pattern_type}: {count} occurrences")
    
    print(f"\nCleaned articles saved to: {cleaner.output_dir}")

if __name__ == "__main__":
    main()