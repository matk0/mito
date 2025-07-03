# Slovak Health GraphRAG Chatbot System

An advanced knowledge graph and chatbot system built from Slovak health expert Jaroslav Lachky's blog content. Features complete GraphRAG implementation with Neo4j knowledge graph, ChromaDB vector search, Rails 8 web interface, and Ollama/Qwen2.5:7b local LLM integration.

## 🎯 Key Features

- **GraphRAG System**: Hybrid vector + knowledge graph retrieval for enhanced accuracy
- **Knowledge Graph**: Neo4j database with 30K+ health entities and their relationships  
- **Entity Extraction**: Custom NER pipeline for Slovak health terminology with linguistic normalization
- **Web Interface**: Rails 8 application with real-time chat interface
- **Local LLM**: Ollama/Qwen2.5:7b integration with OpenAI fallback
- **Slovak Language**: Optimized for Slovak health and quantum biology content
- **Strict Source Fidelity**: LLM responses faithfully present author's views without mainstream corrections

## 🏗️ Complete System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rails 8 Web   │    │   GraphRAG       │    │  Knowledge      │
│   Interface     │◄──►│   Engine         │◄──►│  Graph (Neo4j)  │
│                 │    │                  │    │                 │
│ • Chat UI       │    │ • Vector Search  │    │ • 30K+ entities │
│ • Source links  │    │ • Graph Context  │    │ • Relationships │
│ • Multi-LLM     │    │ • Entity Extract │    │ • Co-occurrence │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Multi-LLM      │
                    │   Integration    │
                    │                  │
                    │ 1. Ollama/Qwen   │
                    │ 2. OpenAI API    │
                    │ 3. Template      │
                    └──────────────────┘
```

## 📊 Knowledge Base Stats

- **184 articles** scraped from jaroslavlachky.sk
- **959 enhanced content chunks** with contextual windows
- **30,380 meaningful entities** extracted and normalized
- **3,178 graph entities** in Neo4j knowledge graph
- **Thousands of relationships** between interconnected health topics
- **604,509 total words** of health and biology content

## 🚀 Quick Start

### 1. Prerequisites

```bash
# System requirements
- Ruby 3.2+
- Python 3.12+
- PostgreSQL
- Node.js
- Docker (for Neo4j)
```

### 2. Install Dependencies

```bash
# Ruby dependencies
cd mito
bundle install
npm install

# Python dependencies  
pip install -r requirements.txt
```

### 3. Set Up Databases

```bash
# PostgreSQL for Rails
rails db:create db:migrate

# Neo4j Knowledge Graph (Docker)
docker run --name neo4j-health -p7474:7474 -p7687:7687 \
  --env NEO4J_AUTH=neo4j/healthgraph123 -d neo4j:5.23

# Build the knowledge graph
python neo4j_graph_builder.py
```

### 4. Set Up Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download Qwen2.5:7b model
ollama pull qwen2.5:7b

# Verify installation
ollama list
```

### 5. Start the Application

```bash
# Start Rails server
cd mito
rails server

# Visit http://localhost:3000
```

## 📁 Complete Project Structure

```
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── NEO4J_SETUP.md                    # Neo4j installation guide
│
├── mito/                              # Rails 8 Web Application
│   ├── app/
│   │   ├── controllers/chat_controller.rb     # Chat API endpoints
│   │   ├── services/intelligent_response_service.rb  # Multi-LLM integration
│   │   └── views/chat/index.html.erb          # Chat interface
│   ├── graphrag_interface.py          # GraphRAG ↔ Rails communication
│   ├── graphrag_system.py            # Core GraphRAG system
│   ├── entity_extractor.py           # Slovak entity extraction
│   └── config/                       # Rails configuration
│
├── Data Processing Pipeline/
│   ├── blog_scraper.rb               # Web scraper for content
│   ├── content_chunker.py            # Enhanced chunking with context windows
│   ├── embedding_generator.py        # Context-aware vector embeddings
│   ├── entity_extractor.py           # Custom Slovak health NER pipeline
│   └── neo4j_graph_builder.py        # Knowledge graph construction
│
├── Data Storage/
│   ├── scraped_data/                 # Raw scraped articles (184 articles)
│   ├── chunked_data/                 # Enhanced chunks + extracted entities
│   │   ├── chunked_content.json      # 959 chunks with context windows
│   │   ├── extracted_entities.json   # 30K+ filtered health entities
│   │   └── entity_analysis.json      # Entity relationships and insights
│   └── vector_db/                    # ChromaDB vector database
│
└── Testing & Analysis/
    ├── test_graphrag.py              # GraphRAG system tests
    ├── test_entity_*.py              # Entity extraction tests
    └── analyze_extracted_entities.py # Entity analysis tools
```

## 🔧 Technical Implementation

### GraphRAG System
- **Vector Search**: Multilingual E5-Large embeddings (1024 dimensions)
- **Knowledge Graph**: Neo4j with entity relationships and co-occurrence patterns
- **Hybrid Retrieval**: Combines semantic similarity with graph traversal
- **Dynamic Scoring**: Vector (similarity) + Graph (relevance) = Combined score
- **Slovak NER**: Custom spaCy pipeline with 15 entity types + linguistic normalization

### Rails Integration
- **Rails 8**: Modern web framework with Hotwire (Turbo + Stimulus)
- **Multi-LLM Support**: Ollama → OpenAI → Template fallback hierarchy
- **GraphRAG Interface**: JSON communication between Rails and GraphRAG system
- **Real-time Chat**: Live chat experience with source attribution
- **Responsive Design**: Mobile-optimized with Tailwind CSS

### AI Integration Strategy
1. **Primary: Ollama/Qwen2.5:7b** (Local, privacy-focused)
2. **Secondary: OpenAI GPT-4o-mini** (Cloud fallback)
3. **Tertiary: Template-based** (Always available)

### Strict Source Fidelity System
- **Specialized Prompts**: Prevents LLM from evaluating or correcting knowledge base
- **Banned Phrases**: Blocks evaluative language ("nie je podložené", "nepravdivé")
- **Role Definition**: LLM acts as "hovorca" (spokesperson), not critic
- **No Disclaimers**: Presents author's views without warnings or corrections

## 📖 Content Topics Covered

The knowledge base covers diverse health and biology topics:

- **Quantum Biology** - Quantum effects in biological systems
- **Mitochondria** - Cellular energy production and health optimization
- **Epigenetics** - Gene expression and environmental factors
- **Hormones** - Endocrine system optimization (leptin, insulin, cortisol)
- **Circadian Rhythms** - Light, sleep, and biological cycles
- **Cold Adaptation** - Thermogenesis and cold therapy protocols
- **Light Therapy** - Photobiomodulation and UV light benefits
- **DHA & Nutrition** - Specialized nutrition and metabolic insights

## 🎯 Use Cases

### Research & Knowledge Discovery
- **Interactive Chat Interface**: Ask questions in Slovak about health topics
- **Source Attribution**: Direct links to original Jaroslav Lachky articles
- **Graph Exploration**: Discover relationships between health concepts
- **Entity Analysis**: Explore interconnected health entities and their relationships

### Advanced Analysis
- **Cypher Queries**: Query the knowledge graph directly via Neo4j Browser
- **Entity Co-occurrence**: Analyze concept relationships across 184 articles
- **Content Exploration**: Navigate complex health topic interconnections

## 🔬 Data Processing Pipeline

1. **Content Acquisition**: Extract 184 articles from jaroslavlachky.sk
2. **Enhanced Chunking**: Create 959 chunks with contextual windows and overlap
3. **Entity Extraction**: Extract 30K+ health entities using custom Slovak NER
4. **Linguistic Normalization**: Merge Slovak variants (mitochondrie → mitochondria)
5. **Relationship Mapping**: Identify co-occurrence patterns and semantic connections
6. **Knowledge Graph Construction**: Build Neo4j graph with normalized entities
7. **Vector Enhancement**: Generate contextual embeddings with source attribution
8. **GraphRAG Integration**: Combine vector and graph databases for hybrid retrieval

## 🚧 Development Status

- ✅ **Complete GraphRAG System**: Hybrid vector + graph retrieval fully implemented
- ✅ **Rails 8 Web Application**: Full chat interface with multi-LLM support
- ✅ **Ollama Integration**: Local Qwen2.5:7b with OpenAI fallback
- ✅ **Slovak Entity Extraction**: Custom NER with linguistic normalization
- ✅ **Strict Source Fidelity**: LLM prompts that faithfully present author's views
- ✅ **Neo4j Knowledge Graph**: 30K+ entities with relationship mapping
- ✅ **Enhanced Vector Database**: ChromaDB with contextual embeddings
- ✅ **Source Attribution**: Direct links to original articles

## 🛠️ Configuration

### Environment Variables (mito/.env)

```bash
# Local LLM Configuration (Primary)
OLLAMA_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen2.5:7b

# OpenAI Configuration (Fallback)
OPENAI_API_KEY=your_api_key_here

# Database
DATABASE_URL=postgresql://localhost/mito_development

# Rails
RAILS_ENV=development
SECRET_KEY_BASE=your_secret_key
```

### Neo4j Configuration
- **URI**: bolt://localhost:7687
- **Username**: neo4j
- **Password**: healthgraph123
- **Browser**: http://localhost:7474

## 🎯 Usage Examples

### Sample Slovak Health Queries
- "Čo sú mitochondrie?" - What are mitochondria?
- "Ako DHA ovplyvňuje zdravie?" - How does DHA affect health?
- "Aký je vplyv chladného šoku na hormóny?" - How does cold shock affect hormones?
- "Čo je kvantová biológia?" - What is quantum biology?

### Expected Response Format
```
Podľa článkov Jaroslava Lachkyho...

[Faithful presentation of author's perspective without evaluation or correction]
```

## 🔍 Neo4j Knowledge Graph Queries

```cypher
// Find most connected health concepts
MATCH (e:Entity)-[:CO_OCCURS]-()
RETURN e.name, e.type, count(*) as connections
ORDER BY connections DESC LIMIT 10

// Explore mitochondria relationships
MATCH (m:Entity {name: 'mitochondria'})-[r:CO_OCCURS]-(connected)
RETURN m, r, connected

// View quantum biology concepts
MATCH (e:Entity)-[r:CO_OCCURS]-(e2)
WHERE e.type = 'PHYSICS_CONCEPT'
RETURN e, r, e2
```

## 📚 Key Resources

- **Web Interface**: http://localhost:3000 (after Rails setup)
- **Neo4j Browser**: http://localhost:7474 (after Neo4j setup)
- **Ollama Interface**: http://localhost:11434 (after Ollama setup)
- **Original Content**: [Jaroslav Lachky's Blog](https://jaroslavlachky.sk)
- **Entity Analysis**: See `chunked_data/entity_analysis.json`

## 🏆 Project Achievements

This project successfully demonstrates:
- **Complete GraphRAG Implementation**: First-class hybrid vector + graph retrieval
- **Multi-Modal AI Integration**: Seamless Ollama + OpenAI + template fallback
- **Slovak Language Optimization**: Custom NER with linguistic normalization  
- **Source Fidelity System**: LLM responses that faithfully present specialized knowledge
- **Production-Ready Architecture**: Rails 8 web app with real-time chat interface
- **Knowledge Graph Excellence**: 30K+ entities with sophisticated relationship mapping
- **Domain Specialization**: Deep integration of Slovak health and quantum biology content

## 🚀 Production Deployment

### Deployment Architecture Options

**Current Development Setup:**
- Rails 8 web application
- Local Ollama/Qwen2.5:7b model
- Neo4j Docker container
- PostgreSQL database
- Python GraphRAG integration via subprocess

**Recommended Production Architectures:**

1. **Containerized Microservices (Recommended)**
   ```
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   Rails     │───▶│  GraphRAG   │───▶│   Neo4j     │
   │   Web App   │    │   API       │    │  Knowledge  │
   │  (Container)│    │ (FastAPI)   │    │   Graph     │
   └─────────────┘    └─────────────┘    └─────────────┘
                             │
                             ▼
                      ┌─────────────┐
                      │   Ollama    │
                      │  Service    │
                      │ (GPU node)  │
                      └─────────────┘
   ```

2. **Alternative: FastAPI + React Frontend**
   - Replace Rails with FastAPI backend + React frontend
   - Direct Python integration (no subprocess calls)
   - Better performance for AI workloads

### Cloud Deployment Considerations

**Infrastructure Requirements:**
- **GPU Instance**: For Ollama/Qwen2.5:7b (8GB+ VRAM recommended)
- **Memory**: 16GB+ RAM for knowledge graph operations
- **Storage**: SSD for vector database performance
- **Network**: Low latency for real-time chat experience

**Platform Options:**
- **AWS/GCP/Azure**: Full control, GPU instances available
- **Railway/Render**: Simplified deployment, good for FastAPI
- **DigitalOcean**: Cost-effective, Docker support
- **Specialized GPU**: RunPod, Vast.ai for Ollama hosting

### Docker Production Setup

```dockerfile
# Dockerfile.rails
FROM ruby:3.2
WORKDIR /app
COPY mito/ .
RUN bundle install
EXPOSE 3000
CMD ["rails", "server", "-b", "0.0.0.0"]

# Dockerfile.graphrag
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py ./
EXPOSE 8000
CMD ["uvicorn", "graphrag_api:app", "--host", "0.0.0.0"]
```

## 🔧 Troubleshooting

### Common Issues

**Neo4j Connection Issues:**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j container
docker restart neo4j-health
```

**Ollama Issues:**
```bash
# Check Ollama status
ollama list
ollama serve

# Pull model if missing
ollama pull qwen2.5:7b
```

**Rails Server Issues:**
```bash
# Check database connection
rails db:setup

# Verify environment
rails console
```

**GraphRAG Integration Issues:**
```bash
# Test GraphRAG interface directly
cd mito
python3.12 graphrag_interface.py --query "test" --format text

# Check Python dependencies
pip install -r requirements.txt

# Verify Neo4j connection
python3.12 -c "from neo4j import GraphDatabase; print('Neo4j OK')"
```

---

**Note**: This system presents health information from a specific author's perspective and should not replace professional medical advice. The system is designed to faithfully present the author's specialized knowledge without mainstream medical corrections or evaluations.