#!/usr/bin/env python3
"""
Article Content Analyzer
Analyzes JSON files from scraped articles to identify content anomalies.
"""

import json
import os
from pathlib import Path
import statistics
from collections import defaultdict
import argparse

class ArticleAnalyzer:
    def __init__(self, articles_dir):
        self.articles_dir = Path(articles_dir)
        self.articles = []
        self.word_counts = []
        self.content_lengths = []
        
    def load_articles(self):
        """Load all JSON articles from the directory"""
        json_files = list(self.articles_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    article['filename'] = file_path.name
                    self.articles.append(article)
                    
                    # Track metrics
                    word_count = article.get('word_count', 0)
                    content_length = len(article.get('content', ''))
                    
                    self.word_counts.append(word_count)
                    self.content_lengths.append(content_length)
                    
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Error reading {file_path}: {e}")
                
        print(f"Loaded {len(self.articles)} articles")
        
    def calculate_statistics(self):
        """Calculate basic statistics for the articles"""
        if not self.word_counts:
            return {}
            
        stats = {
            'total_articles': len(self.articles),
            'word_count_stats': {
                'mean': statistics.mean(self.word_counts),
                'median': statistics.median(self.word_counts),
                'std_dev': statistics.stdev(self.word_counts) if len(self.word_counts) > 1 else 0,
                'min': min(self.word_counts),
                'max': max(self.word_counts),
                'q1': statistics.quantiles(self.word_counts, n=4)[0] if len(self.word_counts) >= 4 else 0,
                'q3': statistics.quantiles(self.word_counts, n=4)[2] if len(self.word_counts) >= 4 else 0,
            },
            'content_length_stats': {
                'mean': statistics.mean(self.content_lengths),
                'median': statistics.median(self.content_lengths),
                'std_dev': statistics.stdev(self.content_lengths) if len(self.content_lengths) > 1 else 0,
                'min': min(self.content_lengths),
                'max': max(self.content_lengths),
            }
        }
        
        return stats
        
    def detect_paywall_indicators(self, article):
        """Detect if an article might be behind a paywall"""
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        
        paywall_indicators = [
            'platených členov',
            'prihláste sa',
            'prihlásiť sa',
            'registrácia',
            'predplatné',
            'premium obsah',
            'prémium obsah',
            'obsah pre predplatiteľov',
            'pokračovanie pre členov',
            'pokračovanie článku',
            'prihlásiť sa na pokračovanie',
            'tento obsah je dostupný iba',
            'obsah je dostupný iba',
            'iba pre registrovaných',
            'login',
            'sign in',
            'register',
            'subscription',
            'premium content',
            'members only',
            'continue reading',
            'read more'
        ]
        
        paywall_score = 0
        found_indicators = []
        
        for indicator in paywall_indicators:
            if indicator in content or indicator in title:
                paywall_score += 1
                found_indicators.append(indicator)
                
        return paywall_score > 0, paywall_score, found_indicators
        
    def detect_suspicious_short_articles(self, stats):
        """Detect articles that are suspiciously short"""
        if not stats:
            return []
            
        # Articles with word count significantly below Q1
        q1_word_count = stats['word_count_stats']['q1']
        threshold = max(50, q1_word_count * 0.3)  # 30% of Q1 or 50 words minimum
        
        short_articles = []
        for article in self.articles:
            word_count = article.get('word_count', 0)
            content_length = len(article.get('content', ''))
            
            if word_count < threshold or content_length < 200:
                short_articles.append({
                    'filename': article['filename'],
                    'title': article.get('title', 'No title'),
                    'word_count': word_count,
                    'content_length': content_length,
                    'url': article.get('url', 'No URL')
                })
                
        return short_articles
        
    def detect_full_articles(self, stats):
        """Detect articles that look like complete blog posts"""
        if not stats:
            return []
            
        # Articles with word count significantly above median
        median_word_count = stats['word_count_stats']['median']
        q3_word_count = stats['word_count_stats']['q3']
        
        # Consider articles above Q3 as likely full articles
        threshold = max(q3_word_count, median_word_count * 1.5)
        
        full_articles = []
        for article in self.articles:
            word_count = article.get('word_count', 0)
            content_length = len(article.get('content', ''))
            
            # Check for good content structure
            has_good_structure = (
                len(article.get('title', '')) > 10 and
                content_length > 1000 and
                word_count > threshold
            )
            
            if has_good_structure:
                full_articles.append({
                    'filename': article['filename'],
                    'title': article.get('title', 'No title'),
                    'word_count': word_count,
                    'content_length': content_length,
                    'url': article.get('url', 'No URL'),
                    'date': article.get('date', 'No date')
                })
                
        return full_articles
        
    def detect_paywall_articles(self):
        """Detect articles that might be behind paywalls"""
        paywall_articles = []
        
        for article in self.articles:
            is_paywall, score, indicators = self.detect_paywall_indicators(article)
            
            if is_paywall:
                paywall_articles.append({
                    'filename': article['filename'],
                    'title': article.get('title', 'No title'),
                    'word_count': article.get('word_count', 0),
                    'content_length': len(article.get('content', '')),
                    'url': article.get('url', 'No URL'),
                    'paywall_score': score,
                    'indicators': indicators
                })
                
        return paywall_articles
        
    def analyze(self):
        """Perform complete analysis"""
        print("Loading articles...")
        self.load_articles()
        
        if not self.articles:
            print("No articles found!")
            return
            
        print("Calculating statistics...")
        stats = self.calculate_statistics()
        
        print("Detecting anomalies...")
        full_articles = self.detect_full_articles(stats)
        paywall_articles = self.detect_paywall_articles()
        short_articles = self.detect_suspicious_short_articles(stats)
        
        return {
            'stats': stats,
            'full_articles': full_articles,
            'paywall_articles': paywall_articles,
            'short_articles': short_articles
        }
        
    def print_results(self, results):
        """Print analysis results"""
        stats = results['stats']
        
        print("\n" + "="*80)
        print("ARTICLE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total articles: {stats['total_articles']}")
        print(f"Word count - Mean: {stats['word_count_stats']['mean']:.1f}, Median: {stats['word_count_stats']['median']:.1f}")
        print(f"Word count - Min: {stats['word_count_stats']['min']}, Max: {stats['word_count_stats']['max']}")
        print(f"Word count - Q1: {stats['word_count_stats']['q1']:.1f}, Q3: {stats['word_count_stats']['q3']:.1f}")
        
        print(f"\nFULL ARTICLES ({len(results['full_articles'])} found):")
        print("-" * 40)
        if results['full_articles']:
            for article in sorted(results['full_articles'], key=lambda x: x['word_count'], reverse=True):
                print(f"• {article['title'][:60]}{'...' if len(article['title']) > 60 else ''}")
                print(f"  Words: {article['word_count']}, Length: {article['content_length']} chars")
                print(f"  File: {article['filename']}")
                print()
        else:
            print("None found")
            
        print(f"\nPAYWALL ARTICLES ({len(results['paywall_articles'])} found):")
        print("-" * 40)
        if results['paywall_articles']:
            for article in sorted(results['paywall_articles'], key=lambda x: x['paywall_score'], reverse=True):
                print(f"• {article['title'][:60]}{'...' if len(article['title']) > 60 else ''}")
                print(f"  Words: {article['word_count']}, Length: {article['content_length']} chars")
                print(f"  Paywall score: {article['paywall_score']}")
                print(f"  Indicators: {', '.join(article['indicators'])}")
                print(f"  File: {article['filename']}")
                print()
        else:
            print("None found")
            
        print(f"\nSUSPICIOUSLY SHORT ARTICLES ({len(results['short_articles'])} found):")
        print("-" * 40)
        if results['short_articles']:
            for article in sorted(results['short_articles'], key=lambda x: x['word_count']):
                print(f"• {article['title'][:60]}{'...' if len(article['title']) > 60 else ''}")
                print(f"  Words: {article['word_count']}, Length: {article['content_length']} chars")
                print(f"  File: {article['filename']}")
                print()
        else:
            print("None found")

def main():
    parser = argparse.ArgumentParser(description='Analyze scraped articles for content anomalies')
    parser.add_argument('--articles-dir', '-d', 
                       default='./data/raw/scraped_data/articles',
                       help='Directory containing article JSON files')
    
    args = parser.parse_args()
    
    analyzer = ArticleAnalyzer(args.articles_dir)
    results = analyzer.analyze()
    
    if results:
        analyzer.print_results(results)

if __name__ == "__main__":
    main()