#!/usr/bin/env python3
"""
Script to analyze JSON files in data/raw/scraped_data/articles/ directory
and output them sorted by word_count along with their URLs.
"""

import json
import os
from pathlib import Path

def main():
    # Path to the articles directory
    articles_dir = Path("data/raw/scraped_data/articles")
    
    if not articles_dir.exists():
        print(f"Directory {articles_dir} does not exist!")
        return
    
    # List to store article data
    articles = []
    
    # Process all JSON files in the directory
    for json_file in articles_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract word count and URL
                word_count = data.get('word_count', 0)
                url = data.get('url', 'No URL')
                filename = json_file.name
                
                articles.append({
                    'filename': filename,
                    'word_count': word_count,
                    'url': url
                })
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # Sort articles by word count (descending order)
    articles.sort(key=lambda x: x['word_count'], reverse=True)
    
    # Output results
    print("Articles sorted by word count (highest to lowest):")
    print("=" * 80)
    
    for i, article in enumerate(articles, 1):
        print(f"{i:3d}. {article['filename']}")
        print(f"     Word count: {article['word_count']:,}")
        print(f"     URL: {article['url']}")
        print()
    
    # Summary statistics
    total_articles = len(articles)
    total_words = sum(article['word_count'] for article in articles)
    avg_words = total_words / total_articles if total_articles > 0 else 0
    
    print("=" * 80)
    print(f"Summary:")
    print(f"Total articles: {total_articles}")
    print(f"Total words: {total_words:,}")
    print(f"Average words per article: {avg_words:.0f}")
    
    if articles:
        print(f"Highest word count: {articles[0]['word_count']:,} ({articles[0]['filename']})")
        print(f"Lowest word count: {articles[-1]['word_count']:,} ({articles[-1]['filename']})")

if __name__ == "__main__":
    main()