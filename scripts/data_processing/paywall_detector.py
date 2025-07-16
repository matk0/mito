#!/usr/bin/env python3
"""
Paywall Detector for Slovak Blog
Finds all articles containing paywall text "Pre dočítanie článku sa musíte PRIHLÁSIŤ"
Based on blog_scraper.py framework
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaywallDetector:
    BASE_URL = 'https://jaroslavlachky.sk'
    BLOG_URL = f'{BASE_URL}/blog/'
    OUTPUT_DIR = './data/raw/scraped_data'
    
    # Paywall text to search for
    PAYWALL_TEXT = "Pre dočítanie článku sa musíte PRIHLÁSIŤ"
    
    def __init__(self):
        self.paywall_articles = []
        self.free_articles = []
        self.failed_urls = []
        self.total_checked = 0
        
        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create output directories"""
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    def fetch_url(self, url):
        """Fetch URL content with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Network error for {url}: {e}")
            return None
    
    def normalize_url(self, url):
        """Normalize relative URLs to absolute URLs"""
        if not url:
            return None
        
        if url.startswith('/'):
            return f"{self.BASE_URL}{url}"
        elif url.startswith('http'):
            return url
        else:
            return f"{self.BASE_URL}/{url}"
    
    def scrape_article_urls_from_page(self, page_num):
        """Scrape article URLs from a blog page"""
        url = self.BLOG_URL if page_num == 1 else f"{self.BLOG_URL}page/{page_num}/"
        
        try:
            response = self.fetch_url(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response, 'html.parser')
            
            # Try multiple selectors to find article links
            selectors = [
                'article h2 a',
                '.entry-title a',
                '.post-title a',
                'h2.entry-title a',
                '.article h2 a'
            ]
            
            article_links = []
            for selector in selectors:
                links = soup.select(selector)
                if links:
                    article_links = [self.normalize_url(link.get('href')) for link in links]
                    break
            
            # Fallback: look for any links that seem like blog posts
            if not article_links:
                all_links = soup.find_all('a', href=True)
                article_links = [
                    self.normalize_url(link['href'])
                    for link in all_links
                    if link['href'] and 
                       self.BASE_URL in link['href'] and 
                       '/blog/' not in link['href'] and 
                       link['href'].count('/') >= 4
                ]
                article_links = list(set(article_links))  # Remove duplicates
            
            return list(set(article_links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error fetching page {page_num}: {e}")
            return []
    
    def extract_title(self, soup):
        """Extract article title"""
        selectors = [
            'h1.entry-title',
            '.entry-title',
            'h1',
            'title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title:
                    return title
        
        return None
    
    def check_for_paywall(self, url):
        """Check if article contains paywall text"""
        response = self.fetch_url(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response, 'html.parser')
        
        # Extract title
        title = self.extract_title(soup)
        if not title:
            title = "Unknown Title"
        
        # Get the full page text
        page_text = soup.get_text()
        
        # Check if paywall text is present
        has_paywall = self.PAYWALL_TEXT in page_text
        
        article_info = {
            'url': url,
            'title': title,
            'has_paywall': has_paywall,
            'checked_at': datetime.now().isoformat()
        }
        
        return article_info
    
    def discover_all_articles(self):
        """Discover all article URLs from the blog"""
        logger.info("Starting article URL discovery...")
        
        all_article_urls = set()
        page = 1
        
        while True:
            logger.info(f"Checking page {page} for article URLs...")
            article_urls = self.scrape_article_urls_from_page(page)
            
            if not article_urls:
                logger.info("No more articles found. Discovery complete.")
                break
            
            new_urls = set(article_urls) - all_article_urls
            all_article_urls.update(new_urls)
            
            logger.info(f"Found {len(article_urls)} URLs on page {page} ({len(new_urls)} new)")
            
            page += 1
            
            # Safety break - adjust if needed
            if page > 50:
                logger.warning("Reached page limit (50). Stopping discovery.")
                break
            
            # Small delay to be respectful
            time.sleep(1)
        
        logger.info(f"Total articles discovered: {len(all_article_urls)}")
        return list(all_article_urls)
    
    def check_all_articles(self):
        """Check all articles for paywall text"""
        logger.info("Starting paywall detection...")
        
        # First, discover all article URLs
        all_urls = self.discover_all_articles()
        
        if not all_urls:
            logger.error("No articles found to check!")
            return
        
        logger.info(f"Checking {len(all_urls)} articles for paywall text...")
        
        for i, url in enumerate(all_urls, 1):
            logger.info(f"Checking article {i}/{len(all_urls)}: {url}")
            
            try:
                article_info = self.check_for_paywall(url)
                if article_info:
                    self.total_checked += 1
                    
                    if article_info['has_paywall']:
                        self.paywall_articles.append(article_info)
                        logger.info(f"✓ PAYWALL FOUND: {article_info['title'][:50]}...")
                    else:
                        self.free_articles.append(article_info)
                        logger.info(f"✓ Free article: {article_info['title'][:50]}...")
                else:
                    logger.warning(f"✗ Failed to check: {url}")
                    self.failed_urls.append(url)
                
                # Small delay to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"✗ Error checking {url}: {e}")
                self.failed_urls.append(url)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate detailed report of paywall detection"""
        logger.info("\n" + "="*80)
        logger.info("PAYWALL DETECTION COMPLETE")
        logger.info("="*80)
        
        logger.info(f"Total articles checked: {self.total_checked}")
        logger.info(f"Articles with paywall: {len(self.paywall_articles)}")
        logger.info(f"Free articles: {len(self.free_articles)}")
        logger.info(f"Failed to check: {len(self.failed_urls)}")
        
        if self.paywall_articles:
            logger.info("\nARTICLES WITH PAYWALL:")
            logger.info("-" * 50)
            for article in self.paywall_articles:
                logger.info(f"  • {article['title']}")
                logger.info(f"    URL: {article['url']}")
                logger.info("")
        
        # Save results to files
        self.save_results()
    
    def save_results(self):
        """Save results to JSON and text files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        report_data = {
            'scan_completed_at': datetime.now().isoformat(),
            'paywall_text_searched': self.PAYWALL_TEXT,
            'total_checked': self.total_checked,
            'paywall_articles_count': len(self.paywall_articles),
            'free_articles_count': len(self.free_articles),
            'failed_urls_count': len(self.failed_urls),
            'paywall_articles': self.paywall_articles,
            'free_articles': self.free_articles,
            'failed_urls': self.failed_urls
        }
        
        json_path = Path(self.OUTPUT_DIR) / f'paywall_detection_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed report saved to: {json_path}")
        
        # Save simple URL list for paywall articles
        if self.paywall_articles:
            urls_path = Path(self.OUTPUT_DIR) / f'paywall_urls_{timestamp}.txt'
            with open(urls_path, 'w', encoding='utf-8') as f:
                f.write("# Articles with paywall text\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Search text: {self.PAYWALL_TEXT}\n\n")
                
                for article in self.paywall_articles:
                    f.write(f"{article['url']}\n")
            
            logger.info(f"Paywall URLs saved to: {urls_path}")
        
        # Save failed URLs if any
        if self.failed_urls:
            failed_path = Path(self.OUTPUT_DIR) / f'paywall_detection_failed_{timestamp}.txt'
            with open(failed_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.failed_urls))
            
            logger.info(f"Failed URLs saved to: {failed_path}")

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Paywall Detector for jaroslavlachky.sk')
    parser.add_argument('--output', '-o', default='./data/raw/scraped_data', 
                       help='Output directory (default: ./data/raw/scraped_data)')
    parser.add_argument('--urls', nargs='*', 
                       help='Check specific URLs instead of scanning entire blog')
    
    args = parser.parse_args()
    
    print("Slovak Blog Paywall Detector")
    print("="*40)
    print(f"Searching for: 'Pre dočítanie článku sa musíte PRIHLÁSIŤ'")
    print("="*40)
    
    # Update output directory if specified
    if args.output:
        PaywallDetector.OUTPUT_DIR = args.output
    
    detector = PaywallDetector()
    
    if args.urls:
        # Check specific URLs
        logger.info(f"Checking {len(args.urls)} specific URLs...")
        for url in args.urls:
            if not url.startswith('http'):
                url = detector.normalize_url(url)
            
            try:
                article_info = detector.check_for_paywall(url)
                if article_info:
                    detector.total_checked += 1
                    
                    if article_info['has_paywall']:
                        detector.paywall_articles.append(article_info)
                        logger.info(f"✓ PAYWALL FOUND: {article_info['title']}")
                    else:
                        detector.free_articles.append(article_info)
                        logger.info(f"✓ Free article: {article_info['title']}")
                else:
                    detector.failed_urls.append(url)
                    logger.warning(f"✗ Failed to check: {url}")
                    
            except Exception as e:
                logger.error(f"✗ Error checking {url}: {e}")
                detector.failed_urls.append(url)
        
        detector.generate_report()
    else:
        # Check entire blog
        detector.check_all_articles()

if __name__ == "__main__":
    main()