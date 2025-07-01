# Slovak Health & Biology Knowledge Chatbot

An intelligent chatbot that answers questions about health, biology, and wellness topics using content from Slovak health expert Jaroslav Lachky's blog. Built using RAG (Retrieval-Augmented Generation) architecture with vector embeddings and semantic search.

## 🎯 Features

- **Semantic Search**: Find relevant content using multilingual embeddings
- **Slovak Language Support**: Optimized for Slovak health and biology content
- **Knowledge Base**: 184 articles covering topics like mitochondria, epigenetics, hormones, nutrition, and quantum biology
- **Vector Database**: ChromaDB with 868 knowledge chunks for precise retrieval
- **Advanced Processing**: Content chunking and embedding generation pipeline

## 📊 Knowledge Base Stats

- **184 articles** scraped from jaroslavlachky.sk
- **868 content chunks** for optimal retrieval
- **604,509 total words** of health and biology content
- Topics include: quantum biology, mitochondria, hormones, nutrition, epigenetics, circadian rhythms

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

# Process and chunk content
python3.12 content_chunker.py

# Generate embeddings and create vector database
python3.12 embedding_generator.py
```

### 3. Run the Chatbot

```bash
# Coming soon - chatbot interface
python3.12 chatbot.py
```

## 🏗️ Architecture

```
User Question → Embedding → Vector Search → Retrieved Context → LLM → Response
```

### Components

1. **Data Pipeline**:
   - `blog_scraper.rb` - Scrapes content from jaroslavlachky.sk
   - `content_chunker.py` - Processes and chunks articles
   - `embedding_generator.py` - Creates vector embeddings

2. **Knowledge Storage**:
   - `scraped_data/` - Raw article data (184 articles)
   - `chunked_data/` - Processed content chunks (868 chunks)
   - `vector_db/` - ChromaDB vector database

3. **Search & Retrieval**:
   - Multilingual E5-Large embeddings (1024 dimensions)
   - ChromaDB for fast similarity search
   - Semantic chunking for optimal context retrieval

## 📁 Project Structure

```
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── blog_scraper.rb          # Web scraper for content
├── content_chunker.py       # Text processing and chunking
├── embedding_generator.py   # Vector embedding generation
├── scraped_data/           # Raw scraped articles
├── chunked_data/           # Processed content chunks
└── vector_db/              # ChromaDB vector database
```

## 🔧 Technical Details

- **Embedding Model**: intfloat/multilingual-e5-large (1.1GB)
- **Vector Database**: ChromaDB with persistent storage
- **Chunking Strategy**: Semantic chunking with overlap
- **Languages**: Python 3.12, Ruby for scraping
- **Content**: Slovak health and biology articles

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

- Get evidence-based answers about health and biology
- Learn about mitochondrial health and optimization
- Understand epigenetic factors affecting wellness
- Explore quantum biology concepts
- Find protocols for health improvement
- Research Slovak health and biology content

## 🔬 Data Processing Pipeline

1. **Scraping**: Extract 184 articles from jaroslavlachky.sk
2. **Chunking**: Split content into 868 semantic chunks
3. **Embedding**: Generate 1024-dimensional vectors
4. **Indexing**: Store in ChromaDB for fast retrieval
5. **Querying**: Semantic search for relevant context

## 🚧 Development Status

- ✅ Content scraping and processing
- ✅ Vector database creation
- ✅ Embedding generation pipeline
- 🔄 Chatbot interface (in progress)
- ⏳ Web interface
- ⏳ Response generation optimization

## 🤝 Contributing

This project focuses on Slovak health and biology content. Contributions welcome for:
- Chatbot interface improvements
- Response quality optimization
- Additional content sources
- Translation capabilities