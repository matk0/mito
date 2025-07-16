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
        
        # Generate detailed report
        self.generate_report()
        
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
    
    def generate_report(self):
        """Generate detailed PDF download report"""
        successful = [r for r in self.download_log if r['status'].startswith('success')]
        already_exists = [r for r in self.download_log if r['status'] == 'already_exists']
        failed = [r for r in self.download_log if r['status'] == 'failed']
        
        # Analyze failure reasons
        paywall_indicators = ['403', '401', 'unauthorized', 'forbidden', 'subscription', 'login']
        network_indicators = ['timeout', 'connection', 'network', 'dns']
        format_indicators = ['not.*pdf', 'html', 'text/html']
        
        paywall_failures = []
        network_failures = []
        format_failures = []
        other_failures = []
        
        for failure in failed:
            url = failure['url']
            # Basic categorization based on common patterns
            if any(domain in url for domain in ['springer.com', 'nature.com', 'sciencedirect.com']):
                paywall_failures.append(failure)
            elif any(domain in url for domain in ['sciencedaily.com', 'centreforbrainhealth.ca', 'afriscitech.com']):
                format_failures.append(failure)  # Likely news/blog sites
            else:
                other_failures.append(failure)
        
        # Generate markdown report
        report_content = f"""# PDF Download Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total URLs processed**: {len(self.download_log)}
- **Successful downloads**: {len(successful)}
- **Already existed**: {len(already_exists)}
- **Failed downloads**: {len(failed)}
- **Success rate**: {(len(successful) / len(self.download_log) * 100):.1f}%

## Successful Downloads
"""
        
        if successful:
            report_content += "| Filename | Source | Status |\n|----------|--------|--------|\n"
            for item in successful:
                source = self.categorize_source(item['url'])
                report_content += f"| {item['filename']} | {source} | {item['status']} |\n"
        else:
            report_content += "No successful downloads.\n"
        
        report_content += "\n## Already Existing Files\n"
        if already_exists:
            report_content += "| Filename | Source |\n|----------|--------|\n"
            for item in already_exists:
                source = self.categorize_source(item['url'])
                report_content += f"| {item['filename']} | {source} |\n"
        else:
            report_content += "No files already existed.\n"
        
        report_content += "\n## Failed Downloads\n"
        if failed:
            report_content += "\n### Likely Paywall/Access Restricted\n"
            if paywall_failures:
                report_content += "| URL | Source | Issue |\n|-----|--------|-------|\n"
                for item in paywall_failures:
                    source = self.categorize_source(item['url'])
                    report_content += f"| {item['url']} | {source} | Paywall/Access restricted |\n"
            else:
                report_content += "None identified.\n"
            
            report_content += "\n### Format Issues (Non-PDF Sources)\n"
            if format_failures:
                report_content += "| URL | Source | Issue |\n|-----|--------|-------|\n"
                for item in format_failures:
                    source = self.categorize_source(item['url'])
                    report_content += f"| {item['url']} | {source} | News/blog site (no PDF) |\n"
            else:
                report_content += "None identified.\n"
            
            report_content += "\n### Other Failures\n"
            if other_failures:
                report_content += "| URL | Source | Issue |\n|-----|--------|-------|\n"
                for item in other_failures:
                    source = self.categorize_source(item['url'])
                    report_content += f"| {item['url']} | {source} | Unknown error |\n"
            else:
                report_content += "None identified.\n"
        else:
            report_content += "No failed downloads.\n"
        
        report_content += f"""

## Analysis

### Source Breakdown
{self.analyze_sources()}

### Recommendations
{self.generate_recommendations()}

### Next Steps
1. For paywall-restricted articles, consider institutional access or alternative sources
2. For news/blog sites, content may need to be scraped as HTML rather than PDF
3. For other failures, manual review may be required

## Technical Details
- Download directory: `{self.output_dir}`
- Delay between requests: {self.delay} seconds
- User agent: Mozilla/5.0 (standard browser simulation)
- Timeout: 30 seconds per download
"""
        
        # Save report
        report_file = self.output_dir / "pdf_download_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Detailed report saved to {report_file}")
    
    def categorize_source(self, url):
        """Categorize the source of a URL"""
        if 'ncbi.nlm.nih.gov/pubmed' in url:
            return 'PubMed'
        elif 'ncbi.nlm.nih.gov/pmc' in url:
            return 'PMC'
        elif 'nature.com' in url:
            return 'Nature'
        elif 'springer.com' in url:
            return 'Springer'
        elif 'sciencedirect.com' in url:
            return 'ScienceDirect'
        elif 'researchgate.net' in url:
            return 'ResearchGate'
        elif 'physicsworld.com' in url:
            return 'Physics World'
        elif 'sciencedaily.com' in url:
            return 'Science Daily'
        elif 'meddeviceonline.com' in url:
            return 'Medical Device Online'
        elif 'sunlightinstitute.org' in url:
            return 'Sunlight Institute'
        elif 'centreforbrainhealth.ca' in url:
            return 'Centre for Brain Health'
        elif 'afriscitech.com' in url:
            return 'AfriSciTech'
        else:
            return 'Other'
    
    def analyze_sources(self):
        """Analyze the distribution of sources"""
        sources = {}
        for item in self.download_log:
            source = self.categorize_source(item['url'])
            if source not in sources:
                sources[source] = {'total': 0, 'successful': 0, 'failed': 0}
            sources[source]['total'] += 1
            if item['status'].startswith('success'):
                sources[source]['successful'] += 1
            elif item['status'] == 'failed':
                sources[source]['failed'] += 1
        
        analysis = "| Source | Total | Successful | Failed | Success Rate |\n"
        analysis += "|--------|-------|------------|--------|-------------|\n"
        for source, stats in sorted(sources.items()):
            rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
            analysis += f"| {source} | {stats['total']} | {stats['successful']} | {stats['failed']} | {rate:.1f}% |\n"
        
        return analysis
    
    def generate_recommendations(self):
        """Generate recommendations based on results"""
        recommendations = []
        
        # Analyze failure patterns
        failed_sources = {}
        for item in self.download_log:
            if item['status'] == 'failed':
                source = self.categorize_source(item['url'])
                failed_sources[source] = failed_sources.get(source, 0) + 1
        
        if 'Springer' in failed_sources or 'Nature' in failed_sources:
            recommendations.append("- Consider institutional access for Springer and Nature articles")
        
        if 'Science Daily' in failed_sources or 'Medical Device Online' in failed_sources:
            recommendations.append("- News/blog sites should be scraped as HTML content, not PDF")
        
        if 'PubMed' in failed_sources:
            recommendations.append("- Some PubMed articles may only have abstracts freely available")
        
        if not recommendations:
            recommendations.append("- No specific recommendations - review individual failures")
        
        return "\n".join(recommendations)

def main():
    """Main function to run the downloader"""
    downloader = PdfDownloader(output_dir="data/raw/pdfs", delay=2)
    
    # Download from articles.md
    articles_file = "config/articles.md"
    if os.path.exists(articles_file):
        results = downloader.download_from_file(articles_file)
        print(f"Downloaded {len([r for r in results if r['status'].startswith('success')])} PDFs")
    else:
        print(f"{articles_file} not found")

if __name__ == "__main__":
    main()