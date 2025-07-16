#!/usr/bin/env python3
"""
Sources Scraper
Extracts references, links, and citations from scraped articles.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SourcesScraper:
    ARTICLES_DIR = './data/raw/scraped_data/articles'
    OUTPUT_DIR = './data/processed/sources'
    
    def __init__(self):
        self.extracted_sources = []
        self.processed_articles = []
        self.failed_articles = []
        self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create output directories"""
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def find_reference_sections(self, content):
        """Find reference sections in article content"""
        # Slovak keywords for references
        reference_keywords = [
            'REFERENCIE',
            'ODKAZY',
            'LINKY',
            'CITÁCIE',
            'ZDROJE',
            'LITERATÚRA',
            'POUŽITÉ ZDROJE',
            'ŠTÚDIE',
            'ODKAZY NA ŠTÚDIE',
            'POUŽITÉ CITÁCIE'
        ]
        
        # Create pattern that matches any combination of these keywords
        pattern = r'(?:' + '|'.join(reference_keywords) + r')(?:\s*,\s*(?:' + '|'.join(reference_keywords) + r'))*\s*:'
        
        # Find the reference section
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Extract everything after the reference section heading
            ref_section = content[match.end():]
            return ref_section.strip()
        
        return None
    
    def extract_urls_from_text(self, text):
        """Extract URLs from text"""
        # Pattern to match URLs
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        
        urls = re.findall(url_pattern, text)
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = url.rstrip('.,;:')
            
            # Basic URL validation
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    cleaned_urls.append(url)
            except:
                continue
        
        return list(set(cleaned_urls))  # Remove duplicates
    
    def categorize_url(self, url):
        """Categorize URL based on domain"""
        domain = urlparse(url).netloc.lower()
        
        # Scientific/academic domains
        if any(keyword in domain for keyword in ['pubmed', 'ncbi', 'doi', 'jstor', 'springer', 'nature', 'science', 'elsevier', 'wiley', 'arxiv', 'plos', 'academic', 'university', 'edu']):
            return 'academic'
        
        # News/magazine domains
        if any(keyword in domain for keyword in ['news', 'magazine', 'journal', 'bbc', 'cnn', 'reuters', 'guardian', 'times', 'post']):
            return 'news'
        
        # Government/official domains
        if any(keyword in domain for keyword in ['gov', 'who', 'fda', 'nih', 'cdc', 'europa.eu']):
            return 'government'
        
        # Social media
        if any(keyword in domain for keyword in ['facebook', 'twitter', 'instagram', 'youtube', 'linkedin']):
            return 'social_media'
        
        # Blog/personal websites
        if any(keyword in domain for keyword in ['blog', 'wordpress', 'medium', 'substack']):
            return 'blog'
        
        return 'other'
    
    def extract_sources_from_article(self, article_file):
        """Extract sources from a single article file"""
        try:
            with open(article_file, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            content = article_data.get('content', '')
            if not content:
                return None
            
            # Find reference section
            ref_section = self.find_reference_sections(content)
            if not ref_section:
                return None
            
            # Extract URLs from reference section
            urls = self.extract_urls_from_text(ref_section)
            if not urls:
                return None
            
            # Categorize URLs
            categorized_sources = {
                'academic': [],
                'news': [],
                'government': [],
                'social_media': [],
                'blog': [],
                'other': []
            }
            
            for url in urls:
                category = self.categorize_url(url)
                categorized_sources[category].append(url)
            
            return {
                'article_file': article_file.name,
                'article_title': article_data.get('title', ''),
                'article_url': article_data.get('url', ''),
                'article_date': article_data.get('date', ''),
                'reference_section': ref_section[:500] + '...' if len(ref_section) > 500 else ref_section,
                'total_sources': len(urls),
                'sources_by_category': categorized_sources,
                'all_sources': urls,
                'extracted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing {article_file}: {e}")
            return None
    
    def save_sources_to_file(self, sources_data):
        """Save extracted sources to individual JSON file"""
        # Create safe filename based on article title
        safe_title = ''.join(c for c in sources_data['article_title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = '_'.join(safe_title.split())[:50]
        filename = f"{safe_title}_sources.json"
        filepath = Path(self.OUTPUT_DIR) / filename
        
        # Ensure unique filename
        counter = 1
        while filepath.exists():
            name_parts = safe_title.split('_')
            if name_parts[-1].isdigit():
                name_parts[-1] = str(counter)
            else:
                name_parts.append(str(counter))
            filename = f"{'_'.join(name_parts)[:50]}_sources.json"
            filepath = Path(self.OUTPUT_DIR) / filename
            counter += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sources_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved sources for: {sources_data['article_title'][:50]}...")
    
    def save_summary_markdown(self):
        """Save extraction summary in markdown format"""
        # Count sources by category
        category_counts = {
            'academic': 0,
            'news': 0,
            'government': 0,
            'social_media': 0,
            'blog': 0,
            'other': 0
        }
        
        total_sources = 0
        unique_domains = set()
        
        for source_data in self.extracted_sources:
            total_sources += source_data['total_sources']
            for category, urls in source_data['sources_by_category'].items():
                category_counts[category] += len(urls)
                for url in urls:
                    unique_domains.add(urlparse(url).netloc.lower())
        
        # Most common domains
        domain_counts = {}
        for source_data in self.extracted_sources:
            for url in source_data['all_sources']:
                domain = urlparse(url).netloc.lower()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Create markdown content
        md_content = f"""# Sources Extraction Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total articles processed**: {len(self.processed_articles)}
- **Articles with sources**: {len(self.extracted_sources)}
- **Articles without sources**: {len(self.processed_articles) - len(self.extracted_sources)}
- **Failed articles**: {len(self.failed_articles)}
- **Total sources found**: {total_sources}
- **Unique domains**: {len(unique_domains)}

## Sources by Category

| Category | Count | Percentage |
|----------|-------|------------|
"""
        
        for category, count in category_counts.items():
            percentage = (count / total_sources * 100) if total_sources > 0 else 0
            md_content += f"| {category.replace('_', ' ').title()} | {count} | {percentage:.1f}% |\n"
        
        md_content += f"""
## Top 20 Domains

| Domain | Count |
|--------|-------|
"""
        
        for domain, count in top_domains:
            md_content += f"| {domain} | {count} |\n"
        
        md_content += f"""
## Articles with Sources

"""
        
        for source_data in self.extracted_sources:
            md_content += f"""### {source_data['article_title']}

- **File**: {source_data['article_file']}
- **URL**: {source_data['article_url']}
- **Date**: {source_data['article_date']}
- **Total Sources**: {source_data['total_sources']}

**Sources by Category:**
"""
            for category, urls in source_data['sources_by_category'].items():
                if urls:
                    md_content += f"- **{category.replace('_', ' ').title()}**: {len(urls)} sources\n"
                    for url in urls:
                        md_content += f"  - {url}\n"
            
            md_content += "\n---\n\n"
        
        if self.failed_articles:
            md_content += f"""## Failed Articles

The following articles failed to process:

"""
            for failed_article in self.failed_articles:
                md_content += f"- {failed_article}\n"
        
        # Save markdown file
        md_path = Path(self.OUTPUT_DIR) / 'sources_extraction_summary.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Markdown summary saved to {md_path}")
    
    def save_summary(self):
        """Save extraction summary in JSON format"""
        # Count sources by category
        category_counts = {
            'academic': 0,
            'news': 0,
            'government': 0,
            'social_media': 0,
            'blog': 0,
            'other': 0
        }
        
        total_sources = 0
        unique_domains = set()
        
        for source_data in self.extracted_sources:
            total_sources += source_data['total_sources']
            for category, urls in source_data['sources_by_category'].items():
                category_counts[category] += len(urls)
                for url in urls:
                    unique_domains.add(urlparse(url).netloc.lower())
        
        # Most common domains
        domain_counts = {}
        for source_data in self.extracted_sources:
            for url in source_data['all_sources']:
                domain = urlparse(url).netloc.lower()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        summary = {
            'total_articles_processed': len(self.processed_articles),
            'articles_with_sources': len(self.extracted_sources),
            'articles_without_sources': len(self.processed_articles) - len(self.extracted_sources),
            'failed_articles': len(self.failed_articles),
            'total_sources_found': total_sources,
            'unique_domains': len(unique_domains),
            'sources_by_category': category_counts,
            'top_domains': top_domains,
            'extraction_completed_at': datetime.now().isoformat()
        }
        
        summary_path = Path(self.OUTPUT_DIR) / 'sources_extraction_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON summary saved to {summary_path}")
    
    def save_failed_articles(self):
        """Save failed articles to file"""
        if self.failed_articles:
            failed_path = Path(self.OUTPUT_DIR) / 'failed_articles.txt'
            with open(failed_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.failed_articles))
            logger.info(f"Failed articles saved to {failed_path}")
    
    def process_all_articles(self):
        """Process all articles in the articles directory"""
        articles_dir = Path(self.ARTICLES_DIR)
        
        if not articles_dir.exists():
            logger.error(f"Articles directory not found: {articles_dir}")
            return
        
        # Get all JSON files
        article_files = list(articles_dir.glob('*.json'))
        
        if not article_files:
            logger.warning(f"No JSON files found in {articles_dir}")
            return
        
        logger.info(f"Found {len(article_files)} articles to process")
        
        # Process each article
        for article_file in article_files:
            try:
                self.processed_articles.append(article_file.name)
                
                sources_data = self.extract_sources_from_article(article_file)
                if sources_data:
                    self.extracted_sources.append(sources_data)
                    self.save_sources_to_file(sources_data)
                    logger.info(f"✓ Extracted {sources_data['total_sources']} sources from: {sources_data['article_title'][:50]}...")
                else:
                    logger.info(f"○ No sources found in: {article_file.name}")
                
            except Exception as e:
                logger.error(f"✗ Error processing {article_file.name}: {e}")
                self.failed_articles.append(article_file.name)
        
        logger.info("\n=== Source Extraction Complete ===")
        logger.info(f"Total articles processed: {len(self.processed_articles)}")
        logger.info(f"Articles with sources: {len(self.extracted_sources)}")
        logger.info(f"Articles without sources: {len(self.processed_articles) - len(self.extracted_sources)}")
        logger.info(f"Failed articles: {len(self.failed_articles)}")
        
        if self.extracted_sources:
            total_sources = sum(s['total_sources'] for s in self.extracted_sources)
            logger.info(f"Total sources extracted: {total_sources}")
        
        self.save_summary()
        self.save_summary_markdown()
        if self.failed_articles:
            self.save_failed_articles()

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Sources Scraper - Extract references from scraped articles')
    parser.add_argument('--articles-dir', '-a', default='./data/raw/scraped_data/articles',
                       help='Articles directory (default: ./data/raw/scraped_data/articles)')
    parser.add_argument('--output', '-o', default='./data/processed/sources',
                       help='Output directory (default: ./data/processed/sources)')
    
    args = parser.parse_args()
    
    print("Sources Scraper")
    print("===============")
    
    # Update directories if specified
    if args.articles_dir:
        SourcesScraper.ARTICLES_DIR = args.articles_dir
    if args.output:
        SourcesScraper.OUTPUT_DIR = args.output
    
    scraper = SourcesScraper()
    scraper.process_all_articles()

if __name__ == "__main__":
    main()