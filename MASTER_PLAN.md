# MASTER PLAN: Hybrid GraphRAG Knowledge Base Implementation

## Executive Summary
Build a multi-modal, multi-lingual knowledge base with advanced GraphRAG capabilities, combining the best of Microsoft GraphRAG's hierarchical clustering with your domain-specific Slovak health expertise, while maintaining strict source traceability.

## Part 1: Architecture Design

### 1.1 Core System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HYBRID GRAPHRAG SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    SOURCE INGESTION LAYER                    │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  Blog Posts ──┐                                            │  │
│  │  Forum ───────┤                                            │  │
│  │  Books/PDFs ──┼─→ Multi-Modal ─→ Entity ─→ Traceability  │  │
│  │  Videos ──────┤    Processor     Extractor   Tracker      │  │
│  │  Sci Papers ──┘                                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    KNOWLEDGE STORAGE LAYER                   │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │  │
│  │  │  ChromaDB   │  │    Neo4j     │  │  Image Store    │   │  │
│  │  │             │  │              │  │                 │   │  │
│  │  │ • Vectors   │  │ • Entities   │  │ • Descriptions  │   │  │
│  │  │ • Metadata  │  │ • Relations  │  │ • Embeddings    │   │  │
│  │  │ • Sources   │  │ • Communities│  │ • Links         │   │  │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    GRAPHRAG ENGINE LAYER                     │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  Query Processor → Entity Extraction → Multi-Search        │  │
│  │         ↓                                   ↓               │  │
│  │  Question Similarity ←─────────────→ Graph Traversal       │  │
│  │         ↓                                   ↓               │  │
│  │  Vector Search ←─────────────────→ Community Context       │  │
│  │         ↓                                   ↓               │  │
│  │  ┌──────────────────────────────────────────────────┐     │  │
│  │  │           HYBRID SCORING & RANKING               │     │  │
│  │  │  • Authority Weighting (Author: 1.0)            │     │  │
│  │  │  • Question Similarity Score                    │     │  │
│  │  │  • Graph Relevance Score                        │     │  │
│  │  │  • Vector Similarity Score                      │     │  │
│  │  │  • Source Type Weighting                        │     │  │
│  │  └──────────────────────────────────────────────────┘     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 RESPONSE GENERATION LAYER                    │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  Context Assembly → Source Attribution → Multi-Modal       │  │
│  │         ↓                    ↓              Response        │  │
│  │  LLM Generation → Citation Linking → Output Formatting     │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Traceability Architecture

```python
# Every piece of content has complete lineage
{
    "content_id": "unique_hash",
    "source_hierarchy": {
        "source_type": "forum|blog|book|video|paper",
        "source_id": "original_source_id",
        "source_metadata": {
            "title": "...",
            "author": "...",
            "date": "...",
            "url": "...",
            "page": 123,  # for books
            "timestamp": "00:15:23",  # for videos
            "thread_id": "...",  # for forums
        },
        "extraction_metadata": {
            "extracted_at": "timestamp",
            "extraction_method": "...",
            "confidence": 0.95
        },
        "chunk_metadata": {
            "chunk_id": "...",
            "position": "start|middle|end",
            "context_window": "..."
        }
    }
}
```

## Part 2: Source-Specific Implementation

### 2.1 Forum Integration (Priority 1)

#### 2.1.1 Forum Scraping Architecture
```python
class ForumScraper:
    def __init__(self):
        self.author_identifier = AuthorIdentifier()
        self.thread_parser = ThreadParser()
        self.qa_extractor = QAExtractor()
    
    def scrape_forum(self):
        # 1. Crawl forum structure
        # 2. Identify threads with author participation
        # 3. Extract hierarchical discussions
        # 4. Preserve thread context
        
    def extract_thread_structure(self, thread):
        return {
            "thread_id": thread.id,
            "title": thread.title,
            "category": thread.category,
            "original_question": {
                "text": question.text,
                "author": question.author,
                "timestamp": question.timestamp,
                "entities": self.extract_entities(question.text)
            },
            "author_responses": [
                {
                    "text": response.text,
                    "timestamp": response.timestamp,
                    "reply_to": response.parent_id,
                    "entities": self.extract_entities(response.text),
                    "authority_score": 1.0  # Author response
                }
            ],
            "community_responses": [...],
            "discussion_flow": self.build_discussion_graph(thread)
        }
```

#### 2.1.2 Question Similarity Engine
```python
class QuestionSimilarityEngine:
    def __init__(self):
        self.question_embedder = SentenceTransformer('slovak-question-model')
        self.semantic_cache = {}
        
    def index_questions(self, forum_data):
        # Extract all questions
        questions = []
        for thread in forum_data:
            questions.append({
                "question": thread["original_question"]["text"],
                "answer": thread["author_responses"][0]["text"] if thread["author_responses"] else None,
                "thread_id": thread["thread_id"],
                "entities": thread["original_question"]["entities"]
            })
        
        # Generate question embeddings
        self.question_embeddings = self.question_embedder.encode(
            [q["question"] for q in questions]
        )
        
    def find_similar_questions(self, user_query, top_k=5):
        query_embedding = self.question_embedder.encode(user_query)
        similarities = cosine_similarity(query_embedding, self.question_embeddings)
        
        # Return ranked similar Q&A pairs with metadata
        return self.rank_by_similarity_and_authority(similarities, top_k)
```

### 2.2 Books with Images (Priority 2)

#### 2.2.1 Enhanced PDF Processing
```python
class MultiModalPDFProcessor:
    def __init__(self):
        self.text_extractor = PDFTextExtractor()
        self.image_extractor = PDFImageExtractor()
        self.image_describer = ImageDescriber()  # GPT-4V or LLaVA
        self.layout_analyzer = LayoutAnalyzer()
        
    def process_book(self, pdf_path):
        book_data = {
            "metadata": self.extract_metadata(pdf_path),
            "chapters": [],
            "images": [],
            "image_text_mappings": []
        }
        
        for page_num, page in enumerate(pdf_pages):
            # Extract text with positional info
            text_blocks = self.text_extractor.extract_with_positions(page)
            
            # Extract images with positions
            images = self.image_extractor.extract_images(page)
            
            for image in images:
                # Generate AI description
                description = self.image_describer.describe(image.data)
                
                # Find surrounding text context
                surrounding_text = self.find_surrounding_text(
                    image.position, text_blocks
                )
                
                # Create rich image entry
                image_entry = {
                    "image_id": f"book_{book_id}_p{page_num}_img{image.index}",
                    "page": page_num,
                    "position": image.position,
                    "ai_description": description,
                    "surrounding_text": surrounding_text,
                    "caption": self.extract_caption(image, text_blocks),
                    "embedding": self.generate_image_embedding(description),
                    "source_attribution": {
                        "book_id": book_id,
                        "page": page_num,
                        "title": book_data["metadata"]["title"]
                    }
                }
                
                book_data["images"].append(image_entry)
                
        return book_data
```

#### 2.2.2 Image Retrieval System
```python
class ImageRetrievalSystem:
    def __init__(self):
        self.image_embeddings = ChromaDB(collection="book_images")
        self.text_to_image_mapper = TextImageMapper()
        
    def index_images(self, book_data):
        for image in book_data["images"]:
            # Index by description
            self.image_embeddings.add(
                documents=[image["ai_description"]],
                metadatas=[image["source_attribution"]],
                ids=[image["image_id"]]
            )
            
            # Map to related text chunks
            self.text_to_image_mapper.link(
                image["image_id"],
                image["surrounding_text"]
            )
    
    def retrieve_relevant_images(self, query, context_chunks):
        # Direct image search
        image_results = self.image_embeddings.query(query, n_results=5)
        
        # Context-based image search
        context_images = []
        for chunk in context_chunks:
            linked_images = self.text_to_image_mapper.get_images(chunk.id)
            context_images.extend(linked_images)
        
        return self.rank_and_merge(image_results, context_images)
```

### 2.3 Video/Audio Transcription (Priority 3)

#### 2.3.1 Transcription Pipeline
```python
class VideoAudioProcessor:
    def __init__(self):
        self.transcriber = WhisperTranscriber(language="sk")
        self.speaker_identifier = SpeakerDiarization()
        self.qa_detector = InterviewQADetector()
        
    def process_media(self, media_path):
        # Extract audio if video
        audio = self.extract_audio(media_path)
        
        # Transcribe with timestamps
        transcript = self.transcriber.transcribe_with_timestamps(audio)
        
        # Identify speakers
        speaker_segments = self.speaker_identifier.diarize(audio)
        
        # Merge transcription with speakers
        annotated_transcript = self.merge_speakers_transcript(
            transcript, speaker_segments
        )
        
        # Detect Q&A patterns
        qa_segments = self.qa_detector.extract_qa_pairs(annotated_transcript)
        
        return {
            "media_id": media_id,
            "transcript": annotated_transcript,
            "qa_segments": qa_segments,
            "author_segments": self.extract_author_segments(annotated_transcript),
            "timeline_index": self.create_timeline_index(annotated_transcript)
        }
    
    def create_timeline_index(self, transcript):
        # Create searchable chunks with precise timestamps
        chunks = []
        for segment in transcript:
            if segment["speaker"] == "jaroslav_lachky":
                chunk = {
                    "text": segment["text"],
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                    "speaker": segment["speaker"],
                    "authority_score": 1.0,
                    "clip_reference": f"{segment['start']}-{segment['end']}",
                    "entities": self.extract_entities(segment["text"])
                }
                chunks.append(chunk)
        return chunks
```

### 2.4 Scientific Papers Enhancement

#### 2.4.1 Cross-Language Entity Linking
```python
class CrossLanguageEntityLinker:
    def __init__(self):
        self.translation_cache = {}
        self.entity_mapper = BilingualEntityMapper()
        
    def link_scientific_to_slovak(self, paper_entities, slovak_entities):
        links = []
        
        for sci_entity in paper_entities:
            # Try direct matching
            direct_match = self.find_direct_match(sci_entity, slovak_entities)
            
            if not direct_match:
                # Try translation
                translated = self.translate_entity(sci_entity)
                translation_match = self.find_match(translated, slovak_entities)
                
            if match:
                links.append({
                    "english_entity": sci_entity,
                    "slovak_entity": match,
                    "confidence": confidence_score,
                    "link_type": "translation|synonym|related"
                })
                
        return links
```

## Part 3: Enhanced GraphRAG Implementation

### 3.1 Hierarchical Clustering (from Microsoft GraphRAG)

```python
class HierarchicalGraphClustering:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.leiden = LeidenAlgorithm()
        
    def build_entity_communities(self):
        # 1. Export entity co-occurrence graph
        with self.driver.session() as session:
            # Get all entity relationships
            query = """
            MATCH (e1:Entity)-[r:CO_OCCURS]->(e2:Entity)
            RETURN e1.name as source, e2.name as target, r.strength as weight
            """
            relationships = session.run(query).data()
        
        # 2. Build NetworkX graph
        G = nx.Graph()
        for rel in relationships:
            G.add_edge(rel['source'], rel['target'], weight=rel['weight'])
        
        # 3. Apply Leiden clustering
        communities = self.leiden.find_communities(G, resolution=1.0)
        
        # 4. Create hierarchical structure
        hierarchy = self.build_hierarchy(communities)
        
        # 5. Store back in Neo4j
        self.store_communities(hierarchy)
        
    def generate_community_summaries(self, use_llm=False):
        # For each community, generate a summary
        with self.driver.session() as session:
            communities = session.run("""
                MATCH (c:Community)<-[:BELONGS_TO]-(e:Entity)
                RETURN c.id as community_id, collect(e.name) as entities
            """).data()
            
        for community in communities:
            if use_llm:
                # Use LLM to generate summary (expensive)
                summary = self.llm_summarize_community(community['entities'])
            else:
                # Use statistical summary (cheap)
                summary = self.statistical_summarize(community['entities'])
            
            self.store_community_summary(community['community_id'], summary)
```

### 3.2 Enhanced Query Processing

```python
class HybridGraphRAGEngine:
    def __init__(self):
        self.vector_search = VectorSearchEngine()
        self.graph_search = GraphSearchEngine()
        self.question_similarity = QuestionSimilarityEngine()
        self.community_search = CommunitySearchEngine()
        
    def process_query(self, query, search_mode="hybrid"):
        # 1. Extract query entities
        query_entities = self.extract_entities(query)
        
        # 2. Determine query type
        query_type = self.classify_query(query)
        
        if query_type == "global" or search_mode == "global":
            # Use community summaries for high-level questions
            return self.global_search(query, query_entities)
            
        elif query_type == "specific" or search_mode == "local":
            # Use focused entity search
            return self.local_search(query, query_entities)
            
        else:  # hybrid mode
            # Combine all approaches
            results = {
                "vector_results": self.vector_search.search(query),
                "graph_results": self.graph_search.traverse(query_entities),
                "similar_questions": self.question_similarity.find_similar(query),
                "community_context": self.community_search.get_context(query_entities)
            }
            
            return self.merge_and_rank_results(results)
    
    def merge_and_rank_results(self, results):
        # Sophisticated ranking algorithm
        all_results = []
        
        for result_type, items in results.items():
            for item in items:
                scored_item = {
                    "content": item.content,
                    "source": item.source,
                    "scores": {
                        "vector_similarity": item.get("similarity_score", 0),
                        "graph_relevance": item.get("graph_score", 0),
                        "question_similarity": item.get("q_similarity", 0),
                        "authority": item.get("authority_score", 0),
                        "source_type_weight": self.get_source_weight(item.source.type)
                    }
                }
                
                # Calculate combined score with dynamic weights
                scored_item["combined_score"] = self.calculate_combined_score(
                    scored_item["scores"]
                )
                
                all_results.append(scored_item)
        
        # Sort by combined score
        return sorted(all_results, key=lambda x: x["combined_score"], reverse=True)
```

### 3.3 Source Attribution System

```python
class SourceAttributionEngine:
    def __init__(self):
        self.source_registry = SourceRegistry()
        self.citation_formatter = CitationFormatter()
        
    def track_source_usage(self, result_item, response_id):
        # Record which sources were used in which responses
        attribution = {
            "response_id": response_id,
            "source_id": result_item["source"]["id"],
            "source_type": result_item["source"]["type"],
            "content_used": result_item["content"],
            "relevance_scores": result_item["scores"],
            "timestamp": datetime.now(),
            "citation": self.citation_formatter.format(result_item["source"])
        }
        
        self.source_registry.record_attribution(attribution)
        return attribution
    
    def generate_response_citations(self, response_id):
        # Get all sources used in this response
        attributions = self.source_registry.get_attributions(response_id)
        
        # Format citations based on source type
        citations = []
        for attr in attributions:
            if attr["source_type"] == "blog":
                citation = f'[{attr["citation"]["title"]}]({attr["citation"]["url"]})'
            elif attr["source_type"] == "forum":
                citation = f'Forum: {attr["citation"]["thread_title"]} - {attr["citation"]["author"]}'
            elif attr["source_type"] == "book":
                citation = f'{attr["citation"]["title"]}, p. {attr["citation"]["page"]}'
            elif attr["source_type"] == "video":
                citation = f'[{attr["citation"]["title"]}]({attr["citation"]["url"]}) at {attr["citation"]["timestamp"]}'
            elif attr["source_type"] == "paper":
                citation = f'{attr["citation"]["authors"]} ({attr["citation"]["year"]}). {attr["citation"]["title"]}'
                
            citations.append({
                "text": citation,
                "source_id": attr["source_id"],
                "relevance": attr["relevance_scores"]["combined_score"]
            })
        
        return sorted(citations, key=lambda x: x["relevance"], reverse=True)
```

## Part 4: Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
1. **Set up MASTER_PLAN.md documentation**
2. **Enhance existing GraphRAG with traceability**
   - Add comprehensive source tracking
   - Implement citation system
   - Create source attribution database

3. **Implement Leiden clustering**
   - Integrate with existing Neo4j
   - Create community detection pipeline
   - Build hierarchical organization

### Phase 2: Forum Integration (Weeks 3-5)
1. **Build forum scraper**
   - Author identification system
   - Thread structure preservation
   - Q&A extraction

2. **Implement question similarity**
   - Train/adapt Slovak question embeddings
   - Build similarity search system
   - Integrate with GraphRAG scoring

3. **Test with sample forum data**
   - Validate author detection
   - Verify Q&A extraction quality
   - Test retrieval accuracy

### Phase 3: Multi-Modal Books (Weeks 6-8)
1. **Enhance PDF processor**
   - Add image extraction
   - Implement layout analysis
   - Build position mapping

2. **Implement image description**
   - Set up GPT-4V/LLaVA pipeline
   - Create description generation
   - Build quality validation

3. **Create image search system**
   - Generate image embeddings
   - Link images to text
   - Build retrieval interface

### Phase 4: Video/Audio (Weeks 9-11)
1. **Set up transcription pipeline**
   - Configure Whisper for Slovak
   - Implement speaker diarization
   - Build timestamp indexing

2. **Create Q&A detection**
   - Pattern matching for interviews
   - Speaker role identification
   - Authority scoring

3. **Build temporal search**
   - Timestamp-based retrieval
   - Clip generation system
   - Integration with GraphRAG

### Phase 5: Integration & Optimization (Weeks 12-14)
1. **Unified search interface**
   - Merge all content types
   - Implement global/local/hybrid modes
   - Optimize performance

2. **Enhanced response generation**
   - Multi-modal response assembly
   - Source preview system
   - Citation management

3. **Testing & refinement**
   - End-to-end testing
   - Performance optimization
   - User acceptance testing

### Phase 6: Production Deployment (Weeks 15-16)
1. **Infrastructure setup**
   - Scale databases
   - Set up monitoring
   - Configure caching

2. **Documentation & training**
   - User documentation
   - Admin guides
   - API documentation

3. **Launch preparation**
   - Final testing
   - Performance benchmarks
   - Go-live checklist

## Part 5: Technical Specifications

### 5.1 Database Schemas

#### Neo4j Schema Enhancement
```cypher
// Communities
CREATE (c:Community {
    id: string,
    level: integer,
    summary: string,
    keywords: list<string>,
    entity_count: integer
})

// Enhanced Entity with source tracking
CREATE (e:Entity {
    name: string,
    normalized_name: string,
    type: string,
    first_seen: datetime,
    sources: list<string>,
    mention_contexts: list<map>
})

// Source nodes
CREATE (s:Source {
    id: string,
    type: string,  // blog|forum|book|video|paper
    title: string,
    author: string,
    url: string,
    authority_score: float
})

// Relationships
(e:Entity)-[:BELONGS_TO]->(c:Community)
(e:Entity)-[:MENTIONED_IN]->(s:Source)
(e1:Entity)-[:CO_OCCURS {strength: float, contexts: list}]->(e2:Entity)
```

#### ChromaDB Collections
```python
collections = {
    "blog_chunks": {
        "embedding_function": "multilingual-e5-large",
        "metadata_fields": ["source_id", "source_type", "author", "date", "url", "position"]
    },
    "forum_questions": {
        "embedding_function": "slovak-question-embedder",
        "metadata_fields": ["thread_id", "author", "timestamp", "answer_available"]
    },
    "book_chunks": {
        "embedding_function": "multilingual-e5-large", 
        "metadata_fields": ["book_id", "chapter", "page", "has_images"]
    },
    "book_images": {
        "embedding_function": "clip-multilingual",
        "metadata_fields": ["book_id", "page", "image_type", "caption"]
    },
    "video_segments": {
        "embedding_function": "multilingual-e5-large",
        "metadata_fields": ["video_id", "speaker", "timestamp", "duration"]
    }
}
```

### 5.2 API Design

```python
class GraphRAGAPI:
    
    @endpoint("/search")
    def search(self, query: str, options: SearchOptions) -> SearchResponse:
        """
        Main search endpoint with full options
        """
        return SearchResponse(
            results=self.graphrag.search(
                query=query,
                mode=options.mode,  # global|local|hybrid
                sources=options.sources,  # filter by source types
                include_images=options.include_images,
                max_results=options.max_results
            ),
            citations=self.get_citations(results),
            metadata={
                "query_entities": extracted_entities,
                "search_mode": options.mode,
                "processing_time": elapsed_time
            }
        )
    
    @endpoint("/similar_questions")
    def find_similar_questions(self, query: str) -> List[QuestionAnswer]:
        """
        Find similar previously answered questions
        """
        return self.question_engine.find_similar(query)
    
    @endpoint("/entity/{entity_name}")
    def get_entity_info(self, entity_name: str) -> EntityInfo:
        """
        Get comprehensive entity information
        """
        return EntityInfo(
            entity=self.graph.get_entity(entity_name),
            community=self.graph.get_entity_community(entity_name),
            related_entities=self.graph.get_related(entity_name),
            source_mentions=self.get_entity_sources(entity_name)
        )
    
    @endpoint("/source/{source_id}")
    def get_source_preview(self, source_id: str) -> SourcePreview:
        """
        Get source preview for citations
        """
        return self.source_manager.get_preview(source_id)
```

### 5.3 Monitoring & Analytics

```python
class GraphRAGMonitoring:
    def __init__(self):
        self.metrics = {
            "query_performance": [],
            "source_usage": defaultdict(int),
            "entity_popularity": defaultdict(int),
            "user_satisfaction": []
        }
    
    def track_query(self, query, results, response_time):
        self.metrics["query_performance"].append({
            "query": query,
            "result_count": len(results),
            "response_time": response_time,
            "timestamp": datetime.now(),
            "sources_used": [r.source_type for r in results]
        })
    
    def track_source_usage(self, source_id, relevance_score):
        self.metrics["source_usage"][source_id] += 1
        
    def generate_analytics_report(self):
        return {
            "avg_response_time": np.mean([q["response_time"] for q in self.metrics["query_performance"]]),
            "popular_entities": sorted(self.metrics["entity_popularity"].items(), key=lambda x: x[1], reverse=True)[:20],
            "source_distribution": dict(self.metrics["source_usage"]),
            "query_patterns": self.analyze_query_patterns()
        }
```

## Part 6: Cost Analysis & Optimization

### 6.1 Cost Breakdown
- **One-time Costs**:
  - Entity extraction: ~$50 (rule-based, minimal LLM)
  - Initial embeddings: ~$100 (one-time generation)
  - Image descriptions: ~$500 (GPT-4V for book images)
  
- **Ongoing Costs**:
  - Query processing: ~$0.01 per query (mostly embedding generation)
  - LLM responses: ~$0.02 per response (using Ollama locally)
  - Optional community summaries: ~$200/month if using LLM

### 6.2 Optimization Strategies
1. **Caching**: Cache question embeddings, entity relationships
2. **Batch Processing**: Process new content in batches
3. **Local Models**: Use Ollama for response generation
4. **Selective LLM Use**: Only use for complex summaries

## Part 7: Success Metrics

1. **Technical Metrics**
   - Query response time < 2 seconds
   - Source attribution accuracy > 99%
   - Entity extraction precision > 90%
   - Cross-modal retrieval accuracy > 85%

2. **Content Coverage**
   - 184 blog posts ✓
   - 500+ forum threads with author
   - 10+ books with images
   - 50+ hours of video/audio
   - 100+ scientific papers

3. **User Satisfaction**
   - Relevant source in top 3 results > 90%
   - Accurate citations 100%
   - Multi-modal responses when appropriate
   - Clear source attribution always

This comprehensive plan integrates the best of Microsoft GraphRAG's innovations while maintaining your domain expertise and adding crucial features like multi-modal support and strict source traceability.