# Slovak Health & Biology Knowledge Graph

An advanced knowledge graph and RAG system built from Slovak health expert Jaroslav Lachky's blog content. Features enhanced chunking, entity extraction, Neo4j knowledge graph, and semantic search for exploring interconnected health concepts from quantum biology to practical wellness protocols.

## 🎯 Key Features

- **Knowledge Graph**: Neo4j graph database with 30K+ health entities and their relationships
- **Entity Extraction**: Custom NER pipeline for Slovak health terminology (hormones, cellular components, diseases, etc.)
- **Enhanced RAG**: Contextual chunking with larger overlap and surrounding context windows
- **Semantic Search**: Multilingual embeddings with improved source attribution
- **Graph Visualization**: Interactive exploration of health concept relationships
- **Slovak Language Support**: Optimized for Slovak health and quantum biology content

## 📊 Knowledge Base Stats

- **184 articles** scraped from jaroslavlachky.sk
- **959 enhanced content chunks** with contextual windows (increased from 868)
- **30,380 meaningful entities** extracted (filtered from 150K+ raw entities)
- **3,760+ unique health concepts** spanning 17 entity categories
- **Thousands of relationships** between interconnected health topics
- **604,509 total words** of health and biology content

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Ruby dependencies (for scraping)
gem install nokogiri
```

### 2. Set Up Knowledge Base

```bash
# Scrape blog content (already done)
ruby blog_scraper.rb

# Enhanced content chunking with contextual windows
python content_chunker.py

# Generate embeddings with enhanced context
python embedding_generator.py

# Extract health entities using custom NER
python entity_extractor.py
```

### 3. Set Up Neo4j Knowledge Graph

```bash
# Install Neo4j (choose one option):

# Option A: Docker (recommended)
docker run --name neo4j-health -p7474:7474 -p7687:7687 \
  --env NEO4J_AUTH=neo4j/healthgraph123 -d neo4j:5.23

# Option B: Download Neo4j Desktop from https://neo4j.com/download/

# Build the knowledge graph
python neo4j_graph_builder.py

# Access Neo4j Browser at http://localhost:7474
```

### 4. Explore the Knowledge Graph

```cypher
// Find most connected health concepts
MATCH (e:Entity)-[:CO_OCCURS]-()
RETURN e.name, e.type, count(*) as connections
ORDER BY connections DESC LIMIT 10

// Explore mitochondria ecosystem
MATCH (m:Entity {name: 'mitochondrie'})-[r:CO_OCCURS]-(connected)
RETURN m, r, connected

// View quantum biology concepts
MATCH (e:Entity)-[r:CO_OCCURS]-(e2)
WHERE e.type = 'PHYSICS_CONCEPT'
RETURN e, r, e2
```

## 🏗️ Architecture

### Enhanced RAG + Knowledge Graph Pipeline

```
User Question → Entity Extraction → Graph Traversal + Vector Search → 
Enhanced Context → Knowledge Graph Relationships → LLM → Response with Citations
```

### Core Components

1. **Enhanced Data Pipeline**:
   - `blog_scraper.rb` - Scrapes content from jaroslavlachky.sk
   - `content_chunker.py` - Enhanced chunking with 200-word overlap + 300-word context windows
   - `embedding_generator.py` - Context-aware embeddings with title + surrounding text
   - `entity_extractor.py` - Custom NER for Slovak health terminology

2. **Knowledge Graph**:
   - `neo4j_graph_builder.py` - Builds comprehensive health knowledge graph
   - **Entities**: 30K+ health concepts (hormones, cellular components, diseases, etc.)
   - **Relationships**: Co-occurrence patterns and semantic connections
   - **Visualization**: Interactive Neo4j Browser interface

3. **Enhanced Storage**:
   - `scraped_data/` - Raw article data (184 articles)
   - `chunked_data/` - Enhanced chunks with contextual metadata (959 chunks)
   - `vector_db/` - ChromaDB vector database with improved embeddings
   - **Neo4j Database** - Graph representation of health knowledge

4. **Multi-Modal Retrieval**:
   - **Vector Search**: Multilingual E5-Large embeddings (1024 dimensions)
   - **Graph Traversal**: Relationship-based concept discovery
   - **Hybrid Approach**: Combines semantic similarity with knowledge graph relationships

## 📁 Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── NEO4J_SETUP.md              # Neo4j installation guide
├── blog_scraper.rb             # Web scraper for content
├── content_chunker.py          # Enhanced chunking with context windows
├── embedding_generator.py      # Context-aware vector embeddings
├── entity_extractor.py         # Custom Slovak health NER pipeline
├── neo4j_graph_builder.py      # Knowledge graph construction
├── analyze_extracted_entities.py  # Entity analysis and insights
├── test_entity_*.py            # Testing and validation scripts
├── run_entity_extraction.py    # Optimized entity extraction runner
├── scraped_data/               # Raw scraped articles (184 articles)
├── chunked_data/               # Enhanced chunks + extracted entities
│   ├── chunked_content.json    # 959 chunks with context windows
│   ├── extracted_entities.json # 30K+ filtered health entities
│   └── entity_analysis.json    # Entity relationships and insights
└── vector_db/                  # ChromaDB vector database
```

## 🔧 Technical Details

### Enhanced Processing
- **Embedding Model**: intfloat/multilingual-e5-large (1.1GB) with contextual enhancement
- **Vector Database**: ChromaDB with enhanced metadata and source attribution
- **Knowledge Graph**: Neo4j with 30K+ entities and relationship mapping
- **Chunking Strategy**: 800-word chunks + 200-word overlap + 300-word context windows
- **Entity Extraction**: Custom spaCy pipeline with 15 Slovak health entity types
- **Filtering**: Advanced noise removal (Slovak stop words, marketing terms)

### Technical Stack
- **Languages**: Python 3.12, Ruby for scraping, Cypher for graph queries
- **Databases**: ChromaDB (vector), Neo4j (graph)
- **NLP**: spaCy, sentence-transformers, custom Slovak health taxonomy
- **Content**: 184 Slovak health articles covering quantum biology to practical protocols

## 📖 Content Topics

The knowledge base covers diverse health and biology topics:

- **Quantum Biology** - Quantum effects in biological systems
- **Mitochondria** - Cellular energy production and health
- **Epigenetics** - Gene expression and environmental factors
- **Hormones** - Endocrine system and hormone optimization
- **Circadian Rhythms** - Light, sleep, and biological cycles
- **Nutrition** - Food, metabolism, and health protocols
- **Cold Adaptation** - Thermogenesis and cold therapy
- **Light Therapy** - Photobiomodulation and light exposure

## 🎯 Use Cases

### Knowledge Discovery
- **Explore interconnected health concepts** through graph visualization
- **Find relationships** between mitochondria, hormones, light therapy, and nutrition
- **Trace knowledge sources** with enhanced attribution and citations
- **Discover concept clusters** and thematic connections

### Research & Analysis  
- **Query complex health relationships** using Cypher graph queries
- **Analyze entity co-occurrence patterns** across 184 expert articles
- **Compare different health approaches** and their connections
- **Research Slovak quantum biology and health optimization content**

## 🔬 Enhanced Data Processing Pipeline

1. **Content Acquisition**: Extract 184 articles from jaroslavlachky.sk
2. **Enhanced Chunking**: Create 959 chunks with contextual windows and overlap
3. **Entity Extraction**: Extract 30K+ health entities using custom Slovak NER
4. **Relationship Mapping**: Identify co-occurrence patterns and semantic connections
5. **Knowledge Graph Construction**: Build Neo4j graph with entities and relationships
6. **Vector Enhancement**: Generate contextual embeddings with source attribution
7. **Multi-Modal Indexing**: Store in both vector database and graph database

## 🚧 Development Status

- ✅ Enhanced content chunking with contextual windows
- ✅ Custom Slovak health entity extraction pipeline
- ✅ Neo4j knowledge graph construction and visualization
- ✅ Advanced entity filtering and noise removal  
- ✅ Enhanced vector embeddings with source attribution
- ✅ Graph-based relationship discovery and analysis
- 🔄 GraphRAG implementation (hybrid vector + graph retrieval)
- ⏳ Advanced chatbot interface with graph-enhanced responses
- ⏳ Web interface for knowledge graph exploration

## 🤝 Contributing

This project focuses on Slovak health and biology content. Contributions welcome for:
- GraphRAG implementation and optimization
- Advanced chatbot interface with graph-enhanced responses
- Knowledge graph expansion and relationship refinement
- Entity extraction improvements for Slovak terminology
- Web interface for interactive graph exploration
- Additional health content sources and integration

## 📚 Key Resources

- **Neo4j Browser**: http://localhost:7474 (after setup)
- **Neo4j Setup Guide**: See `NEO4J_SETUP.md`
- **Entity Analysis**: See `chunked_data/entity_analysis.json`
- **Original Content**: [Jaroslav Lachky's Blog](https://jaroslavlachky.sk)

## 🏆 Project Achievements

This project successfully demonstrates:
- **Advanced entity extraction** from specialized Slovak health content
- **Knowledge graph construction** revealing health concept interconnections  
- **Enhanced RAG pipeline** with contextual chunking and source attribution
- **Sophisticated filtering** removing 79.8% of noise while preserving meaningful entities
- **Interactive visualization** of complex health relationships through Neo4j