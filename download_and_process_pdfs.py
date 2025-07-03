#!/usr/bin/env python3
"""
Complete PDF Download and Processing Pipeline
Downloads PDFs from articles.md and processes them into the existing knowledge base.
"""

import os
import sys
from pathlib import Path
from pdf_downloader import PdfDownloader
from pdf_processor import PdfProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main pipeline function"""
    
    # Check if articles.md exists
    if not os.path.exists('articles.md'):
        logger.error("articles.md not found. Please create it with PDF URLs.")
        sys.exit(1)
    
    # Step 1: Download PDFs
    logger.info("=== Step 1: Downloading PDFs ===")
    downloader = PdfDownloader(output_dir="pdfs", delay=2)
    download_results = downloader.download_from_file("articles.md")
    
    successful_downloads = [r for r in download_results if r['status'].startswith('success')]
    logger.info(f"Successfully downloaded {len(successful_downloads)} PDFs")
    
    # Step 2: Process PDFs
    logger.info("=== Step 2: Processing PDFs ===")
    processor = PdfProcessor(pdf_dir="pdfs", output_dir="scraped_data/pdfs")
    processed_articles = processor.process_all_pdfs()
    
    logger.info(f"Successfully processed {len(processed_articles)} PDFs")
    
    # Step 3: Integration summary
    logger.info("=== Step 3: Integration Summary ===")
    total_words = sum(article['word_count'] for article in processed_articles)
    total_chars = sum(article['char_count'] for article in processed_articles)
    
    print(f"\n📊 PDF Processing Complete!")
    print(f"  📥 Downloaded: {len(successful_downloads)} PDFs")
    print(f"  📄 Processed: {len(processed_articles)} PDFs")
    print(f"  📝 Total words: {total_words:,}")
    print(f"  📊 Total characters: {total_chars:,}")
    print(f"  📁 Output directory: scraped_data/pdfs/")
    
    # Show integration instructions
    print(f"\n🔧 Next Steps for Integration:")
    print(f"  1. Run content_chunker.py on the new PDF articles")
    print(f"  2. Run entity_extractor.py to extract entities")
    print(f"  3. Run embedding_generator.py to create embeddings")
    print(f"  4. Run neo4j_graph_builder.py to update knowledge graph")
    print(f"  5. Update the Rails app to include PDF sources")
    
    # Show sample processed articles
    if processed_articles:
        print(f"\n📚 Sample Processed Articles:")
        for i, article in enumerate(processed_articles[:5]):
            print(f"  {i+1}. {article['filename']} ({article['word_count']} words)")
            if article.get('metadata', {}).get('title'):
                print(f"     Title: {article['metadata']['title']}")

if __name__ == "__main__":
    main()