#!/usr/bin/env python3
import json
import os
import glob

def count_words(text):
    """Count words in a text string"""
    if not text:
        return 0
    return len(text.split())

def analyze_json_files(directory):
    """Analyze all JSON files in directory and return sorted list by word count"""
    results = []
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract content and URL
            content = data.get('content', '')
            url = data.get('url', file_path)
            
            # Count words in content
            word_count = count_words(content)
            
            results.append({
                'url': url,
                'word_count': word_count,
                'filename': os.path.basename(file_path)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Sort by word count (lowest to highest)
    results.sort(key=lambda x: x['word_count'])
    
    return results

if __name__ == "__main__":
    # Use current directory or specify path
    directory = "."
    
    results = analyze_json_files(directory)
    
    print(f"Found {len(results)} JSON files. Sorted by word count (lowest to highest):\n")
    
    for i, item in enumerate(results, 1):
        print(f"{i:3d}. {item['word_count']:4d} words - {item['url']}")
    
    print(f"\nTotal files analyzed: {len(results)}")
    if results:
        print(f"Range: {results[0]['word_count']} - {results[-1]['word_count']} words")