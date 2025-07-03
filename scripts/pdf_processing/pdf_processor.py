#!/usr/bin/env python3
"""
PDF Processor for Slovak Health ChatBot
Processes downloaded PDFs and integrates them into the existing knowledge pipeline.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
import logging
import requests
from typing import List, Dict, Any
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PdfProcessor:
    def __init__(self, pdf_dir="data/raw/pdfs", output_dir="data/raw/scraped_data/pdfs"):
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "extracted_text").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file using multiple methods"""
        
        # Try pdfplumber first (more robust)
        if HAS_PDFPLUMBER:
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text using pdfplumber: {len(text)} chars")
                    return text.strip()
                    
            except Exception as e:
                logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        
        # Fallback to PyPDF2
        if HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        logger.info(f"Successfully extracted text using PyPDF2: {len(text)} chars")
                        return text.strip()
                        
            except Exception as e:
                logger.warning(f"PyPDF2 failed for {pdf_path}: {e}")
        
        # Try reading as plain text (for text-based PDFs)
        try:
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if len(content) > 100:  # Sanity check
                    logger.info(f"Read as text file: {len(content)} chars")
                    return content
        except Exception as e:
            logger.warning(f"Text reading failed for {pdf_path}: {e}")
        
        logger.error(f"All extraction methods failed for {pdf_path}")
        return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\/\\\"\']', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                # Extract basic info
                info = {
                    'filename': pdf_path.name,
                    'file_size': pdf_path.stat().st_size,
                    'page_count': len(pdf_reader.pages),
                    'created_date': datetime.now().isoformat(),
                    'title': metadata.get('/Title', '').strip() if metadata else '',
                    'author': metadata.get('/Author', '').strip() if metadata else '',
                    'subject': metadata.get('/Subject', '').strip() if metadata else '',
                    'creator': metadata.get('/Creator', '').strip() if metadata else '',
                    'producer': metadata.get('/Producer', '').strip() if metadata else ''
                }
                
                # Try to extract DOI or PubMed ID from filename
                filename = pdf_path.stem
                if filename.startswith('pubmed_'):
                    info['pubmed_id'] = filename.replace('pubmed_', '')
                elif filename.startswith('pmc_'):
                    info['pmc_id'] = filename.replace('pmc_', '')
                elif filename.startswith('doi_'):
                    info['doi'] = filename.replace('doi_', '').replace('_', '/')
                
                return info
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'error': str(e)
            }
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file"""
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return None
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Create output structure similar to existing scraped articles
        article_data = {
            'title': metadata.get('title', pdf_path.stem),
            'content': cleaned_text,
            'url': f"file://{pdf_path.absolute()}",
            'source': 'pdf',
            'filename': pdf_path.name,
            'metadata': metadata,
            'scraped_at': datetime.now().isoformat(),
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text)
        }
        
        # Save extracted text
        text_file = self.output_dir / "extracted_text" / f"{pdf_path.stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Save metadata
        metadata_file = self.output_dir / "metadata" / f"{pdf_path.stem}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save article data (compatible with existing pipeline)
        article_file = self.output_dir / f"{pdf_path.stem}.json"
        with open(article_file, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed: {pdf_path.name} ({len(cleaned_text)} chars)")
        return article_data
    
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Process all PDFs in the PDF directory"""
        if not self.pdf_dir.exists():
            logger.error(f"PDF directory not found: {self.pdf_dir}")
            return []
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        processed_articles = []
        
        for pdf_file in pdf_files:
            try:
                article_data = self.process_pdf(pdf_file)
                if article_data:
                    processed_articles.append(article_data)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        # Create processing summary
        summary = {
            'total_pdfs': len(pdf_files),
            'processed_successfully': len(processed_articles),
            'failed': len(pdf_files) - len(processed_articles),
            'total_words': sum(article['word_count'] for article in processed_articles),
            'total_chars': sum(article['char_count'] for article in processed_articles),
            'processing_date': datetime.now().isoformat(),
            'articles': [
                {
                    'filename': article['filename'],
                    'title': article['title'],
                    'word_count': article['word_count']
                }
                for article in processed_articles
            ]
        }
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete: {len(processed_articles)} PDFs processed")
        logger.info(f"Total words: {summary['total_words']}")
        
        return processed_articles

def main():
    """Main function to process PDFs"""
    processor = PdfProcessor(pdf_dir="pdfs", output_dir="scraped_data/pdfs")
    
    # Process all PDFs
    articles = processor.process_all_pdfs()
    
    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {len(articles)} PDFs")
    print(f"  Total words extracted: {sum(a['word_count'] for a in articles)}")
    print(f"  Output directory: {processor.output_dir}")
    
    # Show sample of processed articles
    if articles:
        print(f"\nSample processed articles:")
        for i, article in enumerate(articles[:5]):
            print(f"  {i+1}. {article['filename']} ({article['word_count']} words)")

if __name__ == "__main__":
    main()