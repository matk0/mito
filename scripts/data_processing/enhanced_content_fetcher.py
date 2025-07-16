#!/usr/bin/env python3
"""
Enhanced Content Fetcher
Advanced content fetching with JavaScript detection and improved extraction.
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import Dict, Optional, List, Tuple
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedContentFetcher:
    """Enhanced content fetcher with JS detection and better extraction"""
    
    def __init__(self, delay_between_requests: float = 1.0):
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'sk,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
    def fetch_article(self, url: str) -> Dict:
        """Fetch and analyze a single article"""
        logger.debug(f"Fetching: {url}")
        
        result = {
            'url': url,
            'success': False,
            'fetch_timestamp': datetime.now().isoformat(),
            'title': None,
            'content': None,
            'content_length': 0,
            'word_count': 0,
            'date': None,
            'excerpt': None,
            'js_detected': False,
            'paywall_detected': False,
            'extraction_method': None,
            'dom_analysis': {},
            'errors': []
        }
        
        try:
            # Fetch the page
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                result['errors'].append(f"HTTP {response.status_code}")
                return result
                
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Analyze DOM structure
            result['dom_analysis'] = self._analyze_dom_structure(soup)
            
            # Detect JavaScript-rendered content
            result['js_detected'] = self._detect_js_content(soup, response.text)
            
            # Extract content using multiple strategies
            extraction_results = self._extract_content_multi_strategy(soup)
            
            # Choose best extraction
            best_extraction = self._choose_best_extraction(extraction_results)
            
            if best_extraction:
                result.update(best_extraction)
                result['success'] = True
                result['content_length'] = len(result['content'] or '')
                result['word_count'] = len((result['content'] or '').split())
                
            # Detect paywall
            result['paywall_detected'] = self._detect_paywall(soup, result.get('content', ''))
            
            # Small delay to be respectful
            time.sleep(self.delay)
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            result['errors'].append(str(e))
            
        return result
        
    def _analyze_dom_structure(self, soup: BeautifulSoup) -> Dict:
        """Analyze DOM structure for insights"""
        analysis = {
            'has_article_tag': bool(soup.find('article')),
            'has_main_tag': bool(soup.find('main')),
            'script_count': len(soup.find_all('script')),
            'iframe_count': len(soup.find_all('iframe')),
            'form_count': len(soup.find_all('form')),
            'meta_tags': {},
            'structured_data': False
        }
        
        # Check for common CMS indicators
        cms_indicators = {
            'wordpress': soup.find('meta', {'name': 'generator', 'content': re.compile('WordPress', re.I)}),
            'drupal': soup.find('meta', {'name': 'generator', 'content': re.compile('Drupal', re.I)}),
            'joomla': soup.find('meta', {'name': 'generator', 'content': re.compile('Joomla', re.I)})
        }
        
        for cms, element in cms_indicators.items():
            if element:
                analysis['cms'] = cms
                break
                
        # Extract important meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                analysis['meta_tags'][name] = content
                
        # Check for structured data
        structured_data = soup.find_all('script', {'type': 'application/ld+json'})
        analysis['structured_data'] = len(structured_data) > 0
        
        return analysis
        
    def _detect_js_content(self, soup: BeautifulSoup, raw_html: str) -> bool:
        """Detect if content is likely JavaScript-rendered"""
        indicators = [
            # Common JS framework indicators
            'ng-app',  # Angular
            'data-reactroot',  # React
            'v-app',  # Vue.js
            '__NUXT__',  # Nuxt.js
            'data-server-rendered',  # Vue SSR
            'gatsby',  # Gatsby
        ]
        
        # Check for framework indicators in HTML
        for indicator in indicators:
            if indicator in raw_html:
                return True
                
        # Check for empty content containers with JS
        main_containers = soup.find_all(['main', 'article', '.content', '.post-content'])
        if main_containers:
            for container in main_containers:
                text = container.get_text().strip()
                if len(text) < 100 and container.find_all('script'):
                    return True
                    
        # Check for loading indicators
        loading_indicators = [
            'loading', 'spinner', 'skeleton',
            'načítava', 'načitanie'  # Slovak loading indicators
        ]
        
        for indicator in loading_indicators:
            elements = soup.find_all(class_=re.compile(indicator, re.I))
            if elements:
                return True
                
        return False
        
    def _extract_content_multi_strategy(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract content using multiple strategies"""
        strategies = [
            ('structured_data', self._extract_from_structured_data),
            ('article_tag', self._extract_from_article_tag),
            ('main_content', self._extract_main_content),
            ('content_selectors', self._extract_content_selectors),
            ('fallback', self._extract_fallback_content)
        ]
        
        results = []
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(soup)
                if result and result.get('content'):
                    result['extraction_method'] = strategy_name
                    results.append(result)
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                
        return results
        
    def _extract_from_structured_data(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract from JSON-LD structured data"""
        scripts = soup.find_all('script', {'type': 'application/ld+json'})
        
        for script in scripts:
            try:
                import json
                data = json.loads(script.string)
                
                # Handle arrays
                if isinstance(data, list):
                    data = data[0] if data else {}
                    
                # Look for article data
                if data.get('@type') in ['Article', 'BlogPosting', 'NewsArticle']:
                    return {
                        'title': data.get('headline') or data.get('name'),
                        'content': data.get('articleBody'),
                        'date': data.get('datePublished'),
                        'excerpt': data.get('description')
                    }
            except (json.JSONDecodeError, KeyError):
                continue
                
        return None
        
    def _extract_from_article_tag(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract from HTML5 article tag"""
        article = soup.find('article')
        if not article:
            return None
            
        # Remove unwanted elements
        self._remove_unwanted_elements(article)
        
        # Extract title
        title_selectors = ['h1', '.entry-title', '.post-title', '.article-title']
        title = None
        for selector in title_selectors:
            title_elem = article.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                break
                
        # Extract content
        content_elem = article
        content = content_elem.get_text().strip()
        
        # Extract date
        date_elem = article.find('time')
        date = date_elem.get('datetime') if date_elem else None
        
        return {
            'title': title,
            'content': content,
            'date': date,
            'excerpt': None
        }
        
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract from main content area"""
        main_selectors = [
            'main',
            '.main-content',
            '#main-content',
            '.content-area',
            '.post-content'
        ]
        
        main_elem = None
        for selector in main_selectors:
            main_elem = soup.select_one(selector)
            if main_elem:
                break
                
        if not main_elem:
            return None
            
        # Remove unwanted elements
        self._remove_unwanted_elements(main_elem)
        
        # Extract title
        title_elem = main_elem.find('h1')
        title = title_elem.get_text().strip() if title_elem else None
        
        # Extract content
        content = main_elem.get_text().strip()
        
        return {
            'title': title,
            'content': content,
            'date': None,
            'excerpt': None
        }
        
    def _extract_content_selectors(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract using common content selectors"""
        # Selectors based on the original blog_scraper.py
        content_selectors = [
            '.pf-content .entry_content',
            '.blog_entry_content',
            '.entry_content',
            '.post-content',
            '.article-content',
            '.content'
        ]
        
        title_selectors = [
            'h1.entry-title',
            '.entry-title',
            'h1',
            'title'
        ]
        
        # Extract title
        title = None
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text().strip()
                if title:
                    break
                    
        # Extract content
        content = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove unwanted elements
                self._remove_unwanted_elements(content_elem)
                content = content_elem.get_text().strip()
                if len(content) > 100:
                    break
                    
        return {
            'title': title,
            'content': content,
            'date': None,
            'excerpt': None
        }
        
    def _extract_fallback_content(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Fallback content extraction"""
        # Extract title from various sources
        title = None
        title_sources = [
            soup.find('title'),
            soup.find('h1'),
            soup.find('meta', {'property': 'og:title'}),
            soup.find('meta', {'name': 'twitter:title'})
        ]
        
        for source in title_sources:
            if source:
                if source.name == 'meta':
                    title = source.get('content')
                else:
                    title = source.get_text().strip()
                if title:
                    break
                    
        # Extract content from body, removing navigation and footer
        body = soup.find('body')
        if body:
            # Remove navigation and other non-content elements
            for element in body.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                element.decompose()
                
            content = body.get_text().strip()
        else:
            content = soup.get_text().strip()
            
        return {
            'title': title,
            'content': content,
            'date': None,
            'excerpt': None
        }
        
    def _remove_unwanted_elements(self, element):
        """Remove unwanted elements from content"""
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 'aside',
            '.comments', '.social-share', '.in_share_element',
            '.fb-like', '.twitter-like', '.printfriendly',
            '.ve_form_element', 'form', '.mw_social_icons_container',
            '.related_posts', '.navigation', '.breadcrumb',
            '.advertisement', '.ad', '.ads', '.sidebar'
        ]
        
        for selector in unwanted_selectors:
            for unwanted in element.select(selector):
                unwanted.decompose()
                
    def _choose_best_extraction(self, extractions: List[Dict]) -> Optional[Dict]:
        """Choose the best content extraction"""
        if not extractions:
            return None
            
        # Score each extraction
        scored_extractions = []
        
        for extraction in extractions:
            score = 0
            content = extraction.get('content', '')
            title = extraction.get('title', '')
            
            # Score based on content length
            if content:
                word_count = len(content.split())
                if word_count > 500:
                    score += 3
                elif word_count > 200:
                    score += 2
                elif word_count > 50:
                    score += 1
                    
            # Score based on title presence
            if title and len(title) > 5:
                score += 1
                
            # Score based on date presence
            if extraction.get('date'):
                score += 1
                
            # Prefer certain extraction methods
            method = extraction.get('extraction_method', '')
            if method == 'structured_data':
                score += 2
            elif method == 'article_tag':
                score += 1
                
            scored_extractions.append((score, extraction))
            
        # Return the highest scoring extraction
        scored_extractions.sort(key=lambda x: x[0], reverse=True)
        return scored_extractions[0][1]
        
    def _detect_paywall(self, soup: BeautifulSoup, content: str) -> bool:
        """Detect paywall presence"""
        if not content:
            return False
            
        # Text-based paywall detection
        paywall_texts = [
            'prémium členov', 'premium členov', 'platených členov',
            'registrácia', 'prihlásiť sa', 'predplatné',
            'webinár', 'pokračovanie článku',
            'members only', 'subscription required', 'premium content'
        ]
        
        content_lower = content.lower()
        for text in paywall_texts:
            if text in content_lower:
                return True
                
        # Element-based paywall detection
        paywall_elements = soup.find_all(class_=re.compile(r'(paywall|premium|subscription|member)', re.I))
        if paywall_elements:
            return True
            
        # Check for subscription forms
        subscription_forms = soup.find_all('form', class_=re.compile(r'(subscribe|premium|member)', re.I))
        if subscription_forms:
            return True
            
        return False