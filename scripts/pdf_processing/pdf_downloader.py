#!/usr/bin/env python3
"""
PDF Downloader for Scientific Articles
Downloads PDFs from scientific article URLs and saves them with proper metadata.
"""

import requests
import os
import time
import re
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PdfDownloader:
    def __init__(self, output_dir="data/raw/pdfs", delay=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.download_log = []
        
    def extract_identifier(self, url):
        """Extract article identifier from URL"""
        # PubMed ID
        if 'pubmed' in url:
            match = re.search(r'pubmed/(\d+)', url)
            if match:
                return f"pubmed_{match.group(1)}"
        
        # PMC ID
        if 'pmc/articles' in url:
            match = re.search(r'PMC(\d+)', url)
            if match:
                return f"pmc_{match.group(1)}"
        
        # DOI from URL
        if 'doi.org' in url:
            match = re.search(r'doi\.org/(.+)', url)
            if match:
                return f"doi_{match.group(1).replace('/', '_')}"
        
        # Nature articles
        if 'nature.com' in url:
            match = re.search(r'articles/([^/?]+)', url)
            if match:
                return f"nature_{match.group(1)}"
        
        # Default: use domain + path hash
        parsed = urlparse(url)
        path_hash = abs(hash(parsed.path)) % 10000
        return f"{parsed.netloc.replace('.', '_')}_{path_hash}"
    
    def get_pdf_url(self, url):
        """Try to find direct PDF URL from article page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            
            # Common PDF link patterns
            pdf_patterns = [
                r'href="([^"]+\.pdf[^"]*)"',
                r'href="([^"]+/pdf[^"]*)"',
                r'data-pdf-url="([^"]+)"',
                r'pdf-url="([^"]+)"'
            ]
            
            for pattern in pdf_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if match.startswith('http'):
                        return match
                    elif match.startswith('/'):
                        return f"{urlparse(url).scheme}://{urlparse(url).netloc}{match}"
            
            # PubMed Central PDF links
            if 'pmc/articles' in url:
                pmc_match = re.search(r'PMC(\d+)', url)
                if pmc_match:
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_match.group(1)}/pdf/"
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding PDF URL for {url}: {e}")
            return None
    
    def download_pdf(self, url, filename):
        """Download PDF from URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and len(response.content) < 1000:
                logger.warning(f"Response doesn't appear to be a PDF: {content_type}")
                return False
            
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {filename} ({len(response.content)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return False
    
    def process_url(self, url):
        """Process a single URL - find and download PDF"""
        logger.info(f"Processing: {url}")
        
        identifier = self.extract_identifier(url)
        filename = f"{identifier}.pdf"
        filepath = self.output_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.info(f"Already exists: {filename}")
            return {
                'url': url,
                'filename': filename,
                'status': 'already_exists',
                'timestamp': datetime.now().isoformat()
            }
        
        # Try direct URL first
        if self.download_pdf(url, filename):
            return {
                'url': url,
                'filename': filename,
                'status': 'success_direct',
                'timestamp': datetime.now().isoformat()
            }
        
        # Try to find PDF URL
        pdf_url = self.get_pdf_url(url)
        if pdf_url:
            logger.info(f"Found PDF URL: {pdf_url}")
            if self.download_pdf(pdf_url, filename):
                return {
                    'url': url,
                    'pdf_url': pdf_url,
                    'filename': filename,
                    'status': 'success_indirect',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Failed to download
        return {
            'url': url,
            'filename': filename,
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }
    
    def download_from_file(self, urls_file):
        """Download PDFs from URLs listed in file"""
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Found {len(urls)} URLs to process")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing {i}/{len(urls)}: {url}")
            
            result = self.process_url(url)
            self.download_log.append(result)
            
            # Add delay between requests
            if i < len(urls):
                time.sleep(self.delay)
        
        # Save download log
        log_file = self.output_dir / "download_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
        
        # Print summary
        successful = sum(1 for r in self.download_log if r['status'].startswith('success'))
        already_exists = sum(1 for r in self.download_log if r['status'] == 'already_exists')
        failed = sum(1 for r in self.download_log if r['status'] == 'failed')
        
        logger.info(f"\nDownload Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Already exists: {already_exists}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {len(urls)}")
        
        return self.download_log

def main():
    """Main function to run the downloader"""
    downloader = PdfDownloader(output_dir="pdfs", delay=2)
    
    # Download from articles.md
    if os.path.exists("articles.md"):
        results = downloader.download_from_file("articles.md")
        print(f"Downloaded {len([r for r in results if r['status'].startswith('success')])} PDFs")
    else:
        print("articles.md not found")

if __name__ == "__main__":
    main()