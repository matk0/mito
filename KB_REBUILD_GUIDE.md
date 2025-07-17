# Knowledge Base Rebuild Guide

This guide explains how to recreate the knowledge graph using the cleaned articles and the new versioned system.

## Quick Start: Rebuild from Cleaned Blog Articles

The cleaned articles are located in `data/raw/scraped_data/articles/cleaned_articles/articles/` and contain 179 high-quality Slovak health articles with marketing noise removed.

### 1. Create Knowledge Graph from Cleaned Articles

```bash
# Create a new versioned KB from cleaned blog articles
python scripts/build_knowledge_graph.py \
  --source-type blog \
  --input-dir data/raw/scraped_data/articles/cleaned_articles/articles/ \
  --version-name "cleaned_blog_articles" \
  --description "High-quality blog articles with marketing content removed"
```

This will:
- Create a new version: `v_YYYYMMDD_HHMMSS_cleaned_blog_articles`
- Process all 179 cleaned JSON files
- Generate chunks with enhanced context windows
- Extract Slovak health entities (15 specialized categories)
- Create vector embeddings using multilingual-e5-large
- Build Neo4j knowledge graph with entity relationships
- Save complete provenance metadata

### 2. View Available Versions

```bash
# List all knowledge graph versions
python scripts/build_knowledge_graph.py --list-versions
```

Expected output:
```
📚 Available Knowledge Graph Versions:
============================================================

📦 v_20250117_141530_cleaned_blog_articles (CURRENT)
   Created: 2025-01-17T14:15:30.123456
   Description: High-quality blog articles with marketing content removed
   Sources: blog
     - blog: 179 files, 590,623 words
   Stats: 3,178 entities, 25,000+ relationships
```

### 3. Activate Different Version

```bash
# Switch to a specific version for GraphRAG
python scripts/build_knowledge_graph.py --activate v_20250117_141530_cleaned_blog_articles
```

## Versioned Knowledge Graph Structure

Each version creates a complete, isolated knowledge base:

```
data/knowledge_graphs/
├── v_20250117_141530_cleaned_blog_articles/
│   ├── metadata.json                    # Complete provenance info
│   ├── chunked_data/
│   │   ├── chunked_content.json        # 959+ enhanced chunks
│   │   └── extracted_entities.json    # 30K+ Slovak health entities
│   ├── embeddings/
│   │   └── vector_db/                  # ChromaDB with version-specific collections
│   └── neo4j_export/
│       └── graph_export.cypher        # Backup of graph structure
└── current -> v_20250117_141530_cleaned_blog_articles/  # Symlink to active version
```

## Advanced Usage

### Multiple Source Types (Future)

```bash
# When forum data is available
python scripts/build_knowledge_graph.py \
  --source-types blog,forum \
  --version-name "blog_and_forum_combined"

# When all sources are ready
python scripts/build_knowledge_graph.py \
  --source-types blog,forum,books,videos,papers \
  --version-name "complete_knowledge_base"
```

### Custom Source Paths

```bash
# Use custom paths for different source types
python scripts/build_knowledge_graph.py \
  --source-type books \
  --input-dir /custom/path/to/book/data/ \
  --version-name "custom_books"
```

## Metadata and Traceability

Each version includes complete metadata in `metadata.json`:

```json
{
  "version": "v_20250117_141530_cleaned_blog_articles",
  "created_at": "2025-01-17T14:15:30.123456Z",
  "description": "High-quality blog articles with marketing content removed",
  "sources": {
    "blog": {
      "path": "data/raw/scraped_data/articles/cleaned_articles/articles/",
      "description": "Blog articles from jaroslavlachky.sk",
      "file_count": 179,
      "total_words": 590623
    }
  },
  "processing_stats": {
    "chunks_created": 959,
    "entities_extracted": 30380,
    "graph_entities": 3178,
    "relationships_created": 25000,
    "processing_time_minutes": 45
  },
  "neo4j_database": "neo4j_20250117141530cleanedblogarticles",
  "chromadb_collections": ["v_20250117_141530_cleaned_blog_articles_blog_chunks"]
}
```

## Integration with GraphRAG

The GraphRAG system automatically uses the "current" version:

```python
# GraphRAG reads from the active version
graphrag = SlovakHealthGraphRAG()
results = graphrag.search("Ako DHA ovplyvňuje mitochondrie?")
```

## Troubleshooting

### 1. Missing Dependencies
```bash
# Ensure all dependencies are installed
pip install -r config/requirements.txt
```

### 2. Neo4j Connection
```bash
# Make sure Neo4j is running
docker ps | grep neo4j

# Restart if needed
docker restart neo4j-health
```

### 3. Path Issues
```bash
# Verify cleaned articles exist
ls data/raw/scraped_data/articles/cleaned_articles/articles/*.json | wc -l
# Should show 179 files
```

### 4. Version Conflicts
```bash
# If a version already exists, it will prompt for overwrite
# Use a different version name or confirm overwrite
python scripts/build_knowledge_graph.py \
  --source-type blog \
  --input-dir data/raw/scraped_data/articles/cleaned_articles/articles/ \
  --version-name "cleaned_blog_v2"
```

## Migration from Old System

If you have an existing knowledge graph from the old system:

1. **Create new versioned KB** from cleaned articles (as shown above)
2. **Compare results** using `--list-versions`
3. **Update GraphRAG references** to use new version paths
4. **Archive old data** once satisfied with new version

## Next Steps

This versioned system is designed for the MASTER_PLAN.md multi-source approach:

1. ✅ **Blog articles** (current step)
2. 🔄 **Forum discussions** (next priority)
3. 🔄 **Books with images** (PDF processing + AI descriptions)
4. 🔄 **Video/audio transcripts** (Whisper + speaker identification)
5. 🔄 **Scientific papers** (cross-language entity linking)

Each source type will create new versions that can be combined or used independently.