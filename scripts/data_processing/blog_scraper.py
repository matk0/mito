#!/usr/bin/env python3
"""
Slovak Blog Scraper
Scrapes articles from jaroslavlachky.sk blog.
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

class BlogScraper:
    BASE_URL = 'https://jaroslavlachky.sk'
    BLOG_URL = f'{BASE_URL}/blog/'
    OUTPUT_DIR = './data/raw/scraped_data'
    
    def __init__(self):
        self.scraped_articles = []
        self.failed_urls = []
        
        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create output directories"""
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(f"{self.OUTPUT_DIR}/articles").mkdir(parents=True, exist_ok=True)
    
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
    
    def extract_content(self, soup):
        """Extract article content"""
        selectors = [
            '.pf-content .entry_content',
            '.blog_entry_content',
            '.entry_content',
            '.post-content',
            '.article-content',
            '.content',
            'article .entry-content',
            'main article'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                # Remove unwanted elements
                unwanted_selectors = [
                    'script', 'style', 'nav', 'header', 'footer', 'aside',
                    '.comments', '.social-share', '.in_share_element',
                    '.fb-like', '.twitter-like', '.printfriendly',
                    '.ve_form_element', 'form', '.mw_social_icons_container',
                    '.related_posts'
                ]
                
                for unwanted in unwanted_selectors:
                    for tag in element.select(unwanted):
                        tag.decompose()
                
                content = element.get_text().strip()
                if len(content) > 100:  # Minimum content length
                    return content
        
        # Fallback: try to get main content area
        main_selectors = ['main', '.main', '#main', '#content', '.content-area', '.blog-content']
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                # Remove unwanted elements
                unwanted_selectors = [
                    'script', 'style', 'nav', 'header', 'footer', 'aside',
                    '.comments', '.social-share', '.in_share_element',
                    '.fb-like', '.twitter-like', '.printfriendly', 'form'
                ]
                
                for unwanted in unwanted_selectors:
                    for tag in main_content.select(unwanted):
                        tag.decompose()
                
                content = main_content.get_text().strip()
                if len(content) > 100:
                    return content
        
        return ''
    
    def extract_date(self, soup):
        """Extract article date"""
        selectors = [
            'time[datetime]',
            '.published',
            '.entry-date',
            '.post-date',
            '[class*="date"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get_text().strip()
                try:
                    # Try to parse the date
                    parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return parsed_date.isoformat()
                except:
                    try:
                        # Fallback date parsing
                        from dateutil import parser
                        parsed_date = parser.parse(date_str)
                        return parsed_date.isoformat()
                    except:
                        continue
        
        return None
    
    def extract_excerpt(self, soup):
        """Extract article excerpt"""
        selectors = [
            '.excerpt',
            '.entry-summary',
            '.post-excerpt'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        
        # Fallback: first paragraph of content
        first_p = soup.select_one('.entry-content p, .post-content p, .content p')
        if first_p:
            return first_p.get_text().strip()[:200]
        
        return ''
    
    def scrape_article(self, url):
        """Scrape a single article"""
        response = self.fetch_url(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response, 'html.parser')
        
        # Extract title
        title = self.extract_title(soup)
        if not title:
            return None
        
        # Extract content
        content = self.extract_content(soup)
        if not content:
            return None
        
        # Extract metadata
        date = self.extract_date(soup)
        excerpt = self.extract_excerpt(soup)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'date': date,
            'excerpt': excerpt,
            'scraped_at': datetime.now().isoformat(),
            'word_count': len(content.split())
        }
    
    def save_article_to_file(self, article):
        """Save article to individual JSON file"""
        # Create safe filename
        safe_title = ''.join(c for c in article['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = '_'.join(safe_title.split())[:50]
        filename = f"{safe_title}.json"
        filepath = Path(self.OUTPUT_DIR) / 'articles' / filename
        
        # Ensure unique filename
        counter = 1
        while filepath.exists():
            name_parts = safe_title.split('_')
            if name_parts[-1].isdigit():
                name_parts[-1] = str(counter)
            else:
                name_parts.append(str(counter))
            filename = f"{'_'.join(name_parts)[:50]}.json"
            filepath = Path(self.OUTPUT_DIR) / 'articles' / filename
            counter += 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
    
    def save_summary(self):
        """Save scraping summary"""
        # Group articles by year
        articles_by_year = {}
        for article in self.scraped_articles:
            if article.get('date'):
                try:
                    year = datetime.fromisoformat(article['date'].replace('Z', '+00:00')).year
                    articles_by_year[year] = articles_by_year.get(year, 0) + 1
                except:
                    articles_by_year['unknown'] = articles_by_year.get('unknown', 0) + 1
            else:
                articles_by_year['unknown'] = articles_by_year.get('unknown', 0) + 1
        
        summary = {
            'total_articles': len(self.scraped_articles),
            'scraping_completed_at': datetime.now().isoformat(),
            'total_words': sum(article['word_count'] for article in self.scraped_articles),
            'articles_by_year': articles_by_year
        }
        
        summary_path = Path(self.OUTPUT_DIR) / 'scraping_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_path}")
    
    def save_failed_urls(self):
        """Save failed URLs to file"""
        if self.failed_urls:
            failed_path = Path(self.OUTPUT_DIR) / 'failed_urls.txt'
            with open(failed_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.failed_urls))
            logger.info(f"Failed URLs saved to {failed_path}")
    
    def scrape_all(self):
        """Main scraping function"""
        logger.info(f"Starting to scrape {self.BLOG_URL}")
        
        page = 1
        total_scraped = 0
        
        while True:
            logger.info(f"\n--- Scraping page {page} ---")
            article_urls = self.scrape_article_urls_from_page(page)
            
            if not article_urls:
                logger.info("No more articles found. Stopping.")
                break
            
            logger.info(f"Found {len(article_urls)} articles on page {page}")
            
            for url in article_urls:
                try:
                    article_data = self.scrape_article(url)
                    if article_data:
                        self.scraped_articles.append(article_data)
                        total_scraped += 1
                        logger.info(f"✓ Scraped: {article_data['title'][:50]}...")
                        
                        # Save individual article
                        self.save_article_to_file(article_data)
                        
                        # Small delay to be respectful
                        time.sleep(0.5)
                    else:
                        logger.warning(f"✗ Failed to scrape: {url}")
                        self.failed_urls.append(url)
                
                except Exception as e:
                    logger.error(f"✗ Error scraping {url}: {e}")
                    self.failed_urls.append(url)
            
            page += 1
            
            # Safety break - adjust if needed
            if page > 50:
                break
        
        logger.info("\n=== Scraping Complete ===")
        logger.info(f"Total articles scraped: {total_scraped}")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")
        
        self.save_summary()
        if self.failed_urls:
            self.save_failed_urls()

def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Slovak Blog Scraper for jaroslavlachky.sk')
    parser.add_argument('--output', '-o', default='./data/raw/scraped_data', 
                       help='Output directory (default: ./data/raw/scraped_data)')
    
    args = parser.parse_args()
    
    print("Slovak Blog Scraper")
    print("===================")
    
    # Update output directory if specified
    if args.output:
        BlogScraper.OUTPUT_DIR = args.output
    
    scraper = BlogScraper()
    
    scraper.scrape_all()

if __name__ == "__main__":
    main()