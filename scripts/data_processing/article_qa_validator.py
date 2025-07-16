#!/usr/bin/env python3
"""
Article Quality Assurance Validator
Comprehensive QA system for validating scraped blog content quality.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArticleQAValidator:
    """Main QA validator that orchestrates all quality checks"""
    
    def __init__(self, articles_dir: str, output_dir: str = None):
        self.articles_dir = Path(articles_dir)
        self.output_dir = Path(output_dir) if output_dir else self.articles_dir / "qa_reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.qa_results = []
        self.summary_stats = {
            'total_articles': 0,
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0,
            'suspicious_articles': 0,
            'analysis_start_time': None,
            'analysis_end_time': None
        }
        
        # Quality thresholds (configurable)
        self.thresholds = {
            'min_words_critical': 50,
            'min_words_major': 200,
            'content_similarity_threshold': 0.7,
            'length_ratio_threshold': 0.8,
            'paywall_confidence_threshold': 0.8
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
        self.summary_stats['total_articles'] = len(articles)
        return articles
        
    def analyze_article(self, article: Dict) -> Dict:
        """Perform comprehensive analysis on a single article"""
        result = {
            'url': article.get('url', 'unknown'),
            'title': article.get('title', 'unknown'),
            'source_file': article.get('source_file', 'unknown'),
            'original_word_count': article.get('word_count', 0),
            'issues': [],
            'warnings': [],
            'quality_score': 0.0,
            'graphrag_suitable': False,
            'issue_category': 'unknown',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Basic content validation
        content = article.get('content', '')
        word_count = len(content.split()) if content else 0
        
        # Check for critical issues
        if word_count < self.thresholds['min_words_critical']:
            result['issues'].append({
                'type': 'critical',
                'description': f'Extremely short content ({word_count} words)',
                'confidence': 1.0
            })
            result['issue_category'] = 'critical'
            self.summary_stats['critical_issues'] += 1
            
        # Check for paywall indicators
        paywall_score = self._detect_paywall_content(content)
        if paywall_score > self.thresholds['paywall_confidence_threshold']:
            result['issues'].append({
                'type': 'critical',
                'description': f'Paywall content detected (confidence: {paywall_score:.2f})',
                'confidence': paywall_score
            })
            if result['issue_category'] != 'critical':
                result['issue_category'] = 'critical'
                self.summary_stats['critical_issues'] += 1
                
        # Check for major issues
        if self.thresholds['min_words_critical'] <= word_count < self.thresholds['min_words_major']:
            result['issues'].append({
                'type': 'major',
                'description': f'Short content ({word_count} words)',
                'confidence': 0.8
            })
            if result['issue_category'] not in ['critical']:
                result['issue_category'] = 'major'
                self.summary_stats['major_issues'] += 1
                
        # Content coherence check
        coherence_issues = self._check_content_coherence(content)
        if coherence_issues:
            result['warnings'].extend(coherence_issues)
            
        # Metadata validation
        metadata_issues = self._validate_metadata(article)
        if metadata_issues:
            result['warnings'].extend(metadata_issues)
            
        # Calculate quality score
        result['quality_score'] = self._calculate_quality_score(article, result)
        
        # Determine GraphRAG suitability
        result['graphrag_suitable'] = (
            result['quality_score'] > 0.6 and
            result['issue_category'] not in ['critical'] and
            word_count >= self.thresholds['min_words_major']
        )
        
        # Set final category if not already set
        if result['issue_category'] == 'unknown':
            if result['warnings']:
                result['issue_category'] = 'minor'
                self.summary_stats['minor_issues'] += 1
            else:
                result['issue_category'] = 'good'
                
        return result
        
    def _detect_paywall_content(self, content: str) -> float:
        """Detect paywall/premium content indicators"""
        if not content:
            return 0.0
            
        content_lower = content.lower()
        
        # Slovak paywall indicators
        paywall_indicators = [
            'prémium členov', 'premium členov', 'platených členov',
            'registrácia', 'registracia', 'prihlásiť sa', 'prihlásenie',
            'predplatné', 'predplatne', 'webinár', 'webinar',
            'pokračovanie článku', 'pokracovanie clanku',
            'obsah pre predplatiteľov', 'obsah pre predplatitelov',
            'pridaj sa medzi', 'this content is for',
            'members only', 'premium content', 'subscription required'
        ]
        
        # Weight-based scoring
        score = 0.0
        total_weight = 0.0
        
        for indicator in paywall_indicators:
            if indicator in content_lower:
                weight = 1.0
                # Higher weight for stronger indicators
                if any(strong in indicator for strong in ['premium', 'predplatne', 'members only']):
                    weight = 2.0
                score += weight
                total_weight += weight
                
        # Check for content truncation patterns
        truncation_patterns = [
            'pokračovanie nájdete', 'pokracovanie najdete',
            'zvyšok článku', 'zvysok clanku',
            'čítať viac', 'citat viac',
            'read more', 'continue reading'
        ]
        
        for pattern in truncation_patterns:
            if pattern in content_lower:
                score += 1.5
                total_weight += 1.5
                
        # Normalize score
        if total_weight > 0:
            return min(score / 3.0, 1.0)  # Normalize to 0-1
        return 0.0
        
    def _check_content_coherence(self, content: str) -> List[Dict]:
        """Check for content coherence issues"""
        issues = []
        
        if not content:
            return [{'type': 'warning', 'description': 'Empty content', 'confidence': 1.0}]
            
        # Check for unusual content patterns
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 3:
            issues.append({
                'type': 'warning',
                'description': 'Very few content lines',
                'confidence': 0.7
            })
            
        # Check for excessive repetition
        words = content.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
            issues.append({
                'type': 'warning',
                'description': 'High word repetition detected',
                'confidence': 0.6
            })
            
        return issues
        
    def _validate_metadata(self, article: Dict) -> List[Dict]:
        """Validate article metadata"""
        issues = []
        
        # Check required fields
        required_fields = ['url', 'title', 'content', 'date']
        for field in required_fields:
            if not article.get(field):
                issues.append({
                    'type': 'warning',
                    'description': f'Missing {field}',
                    'confidence': 1.0
                })
                
        # Validate URL format
        url = article.get('url', '')
        if url and not url.startswith('http'):
            issues.append({
                'type': 'warning',
                'description': 'Invalid URL format',
                'confidence': 0.8
            })
            
        # Check title length
        title = article.get('title', '')
        if len(title) < 10:
            issues.append({
                'type': 'warning',
                'description': 'Very short title',
                'confidence': 0.7
            })
        elif len(title) > 200:
            issues.append({
                'type': 'warning',
                'description': 'Unusually long title',
                'confidence': 0.6
            })
            
        return issues
        
    def _calculate_quality_score(self, article: Dict, analysis_result: Dict) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Penalize based on issues
        for issue in analysis_result['issues']:
            if issue['type'] == 'critical':
                score -= 0.4 * issue['confidence']
            elif issue['type'] == 'major':
                score -= 0.2 * issue['confidence']
                
        # Penalize based on warnings
        for warning in analysis_result['warnings']:
            score -= 0.1 * warning['confidence']
            
        # Bonus for good content length
        word_count = article.get('word_count', 0)
        if word_count > 1000:
            score += 0.1
        elif word_count > 500:
            score += 0.05
            
        # Bonus for metadata completeness
        required_fields = ['url', 'title', 'content', 'date', 'excerpt']
        complete_fields = sum(1 for field in required_fields if article.get(field))
        score += (complete_fields / len(required_fields)) * 0.1
        
        return max(0.0, min(1.0, score))
        
    def run_analysis(self) -> Dict:
        """Run complete QA analysis"""
        logger.info("Starting QA analysis...")
        self.summary_stats['analysis_start_time'] = datetime.now().isoformat()
        
        articles = self.load_articles()
        
        # Analyze each article
        for i, article in enumerate(articles):
            logger.info(f"Analyzing article {i+1}/{len(articles)}: {article.get('title', 'unknown')[:50]}...")
            result = self.analyze_article(article)
            self.qa_results.append(result)
            
        self.summary_stats['analysis_end_time'] = datetime.now().isoformat()
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Save results
        self._save_results()
        
        logger.info(f"QA analysis complete. Found {self.summary_stats['critical_issues']} critical issues")
        return summary
        
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        summary = {
            'overview': self.summary_stats.copy(),
            'quality_distribution': {},
            'issue_breakdown': {},
            'graphrag_ready': 0,
            'recommendations': []
        }
        
        # Quality score distribution
        quality_ranges = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in quality_ranges:
            count = sum(1 for r in self.qa_results if low <= r['quality_score'] < high)
            summary['quality_distribution'][f'{low}-{high}'] = count
            
        # GraphRAG ready count
        summary['graphrag_ready'] = sum(1 for r in self.qa_results if r['graphrag_suitable'])
        
        # Issue breakdown by type
        issue_types = {}
        for result in self.qa_results:
            for issue in result['issues']:
                issue_type = issue['type']
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                
        summary['issue_breakdown'] = issue_types
        
        # Generate recommendations
        critical_count = self.summary_stats['critical_issues']
        total_count = self.summary_stats['total_articles']
        
        if critical_count > total_count * 0.2:
            summary['recommendations'].append(
                "High number of critical issues detected. Consider improving scraping strategy."
            )
            
        if summary['graphrag_ready'] < total_count * 0.7:
            summary['recommendations'].append(
                "Low GraphRAG readiness. Focus on re-scraping problematic articles."
            )
            
        return summary
        
    def _save_results(self):
        """Save analysis results to files"""
        # Save detailed results
        results_file = self.output_dir / f"qa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.summary_stats,
                'results': self.qa_results
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {results_file}")
        
    def get_critical_articles(self) -> List[Dict]:
        """Get list of articles with critical issues"""
        return [r for r in self.qa_results if r['issue_category'] == 'critical']
        
    def get_graphrag_ready_articles(self) -> List[Dict]:
        """Get list of articles ready for GraphRAG"""
        return [r for r in self.qa_results if r['graphrag_suitable']]

def main():
    parser = argparse.ArgumentParser(description='Article Quality Assurance Validator')
    parser.add_argument('--articles-dir', '-d', 
                       default='./data/raw/scraped_data/articles',
                       help='Directory containing article JSON files')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for QA reports')
    parser.add_argument('--threshold-min-words', type=int, default=50,
                       help='Minimum word count threshold for critical issues')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ArticleQAValidator(args.articles_dir, args.output_dir)
    
    # Update thresholds if provided
    if args.threshold_min_words:
        validator.thresholds['min_words_critical'] = args.threshold_min_words
        
    # Run analysis
    summary = validator.run_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("ARTICLE QA ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total articles analyzed: {summary['overview']['total_articles']}")
    print(f"Critical issues: {summary['overview']['critical_issues']}")
    print(f"Major issues: {summary['overview']['major_issues']}")
    print(f"Minor issues: {summary['overview']['minor_issues']}")
    print(f"GraphRAG ready articles: {summary['graphrag_ready']}")
    print(f"GraphRAG readiness: {summary['graphrag_ready']/summary['overview']['total_articles']*100:.1f}%")
    
    if summary['recommendations']:
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"- {rec}")
            
    # Show critical articles
    critical = validator.get_critical_articles()
    if critical:
        print(f"\nCRITICAL ARTICLES ({len(critical)} found):")
        for article in critical[:10]:  # Show first 10
            print(f"- {article['title'][:60]}... ({article['original_word_count']} words)")
            
    print(f"\nDetailed results saved to: {validator.output_dir}")

if __name__ == "__main__":
    main()