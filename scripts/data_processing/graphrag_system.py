#!/usr/bin/env python3

"""
GraphRAG System for Slovak Health Knowledge Graph
Combines vector similarity search with knowledge graph traversal for enhanced retrieval.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

# Database and ML imports
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import networkx as nx

# Local imports
import sys
sys.path.append(str(Path(__file__).parent))

@dataclass
class GraphRAGResult:
    """Represents a GraphRAG retrieval result."""
    content: str
    source_url: str
    source_title: str
    similarity_score: float
    graph_relevance_score: float
    combined_score: float
    related_entities: List[str]
    entity_relationships: List[Dict[str, Any]]
    chunk_metadata: Dict[str, Any]

@dataclass
class EntityContext:
    """Context about an entity from the knowledge graph."""
    entity_name: str
    entity_type: str
    mention_count: int
    related_entities: List[Tuple[str, float]]  # (entity, strength)
    community_id: Optional[int] = None

class SlovakHealthGraphRAG:
    """
    GraphRAG system for Slovak health content.
    Combines ChromaDB vector search with Neo4j knowledge graph traversal.
    """
    
    def __init__(self,
                 # Vector database config
                 chroma_db_path: str = "./data/embeddings/vector_db",
                 chroma_collection: str = "slovak_blog_chunks",
                 embedding_model: str = "intfloat/multilingual-e5-large",
                 
                 # Graph database config  
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j",
                 neo4j_password: str = "healthgraph123"):
        """
        Initialize GraphRAG system.
        
        Args:
            chroma_db_path: Path to ChromaDB vector database
            chroma_collection: ChromaDB collection name
            embedding_model: SentenceTransformer model name
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
        """
        self.chroma_db_path = chroma_db_path
        self.chroma_collection = chroma_collection
        self.embedding_model_name = embedding_model
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Initialize components
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_collection_obj = None
        self.neo4j_driver = None
        
        # Entity extractor (reuse from existing system)
        self.entity_extractor = None
        
        # Cache for performance
        self.entity_cache = {}
        self.community_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        print("🚀 Initializing GraphRAG System...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Load embedding model
        print("🤖 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Connect to ChromaDB
        print("📊 Connecting to ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.chroma_collection_obj = self.chroma_client.get_collection(self.chroma_collection)
        
        # Connect to Neo4j
        print("🌐 Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Test connections
        self._test_connections()
        
        # Initialize entity extractor
        print("🔍 Initializing entity extractor...")
        try:
            from entity_extractor import SlovakHealthEntityExtractor
            self.entity_extractor = SlovakHealthEntityExtractor()
        except Exception as e:
            print(f"⚠️  Could not load entity extractor: {e}")
        
        print("✅ GraphRAG system initialized successfully!")
    
    def _test_connections(self):
        """Test database connections."""
        # Test ChromaDB
        try:
            count = self.chroma_collection_obj.count()
            print(f"   📊 ChromaDB: {count} chunks available")
        except Exception as e:
            raise ConnectionError(f"ChromaDB connection failed: {e}")
        
        # Test Neo4j
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (e:Entity) RETURN count(e) as entity_count").single()
                entity_count = result["entity_count"]
                print(f"   🌐 Neo4j: {entity_count} entities available")
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {e}")
    
    def extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from user query."""
        if not self.entity_extractor:
            return []
        
        try:
            entities = self.entity_extractor.extract_entities(query)
            # Filter and return just the entity text, excluding common words
            entity_texts = []
            for entity in entities:
                if (not entity.label.startswith('SPACY_') and 
                    not entity.label == 'CHEMICAL_FORMULA' and 
                    len(entity.text) > 2):
                    entity_texts.append(entity.text.lower().strip())
            
            # Add Slovak word stem matching for better entity recognition
            entity_texts.extend(self._extract_slovak_stems(query))
            
            # Remove duplicates and short words
            entity_texts = list(set([t for t in entity_texts if len(t) > 2]))
            return entity_texts
        except Exception as e:
            print(f"⚠️  Entity extraction error: {e}")
            return []
    
    def _extract_slovak_stems(self, query: str) -> List[str]:
        """Extract Slovak word stems that might be entities."""
        stems = []
        
        # Slovak-specific mappings for common health terms
        slovak_mappings = {
            'chladný': 'chlad', 'chladného': 'chlad', 'chladnom': 'chlad',
            'hormóny': 'hormóny', 'hormónov': 'hormóny', 'hormónom': 'hormóny',
            'mitochondrie': 'mitochondrie', 'mitochondrií': 'mitochondrie', 
            'mitochondriám': 'mitochondrie', 'mitochondriach': 'mitochondrie',
            'energiou': 'energia', 'energie': 'energia', 'energii': 'energia',
            'svetlo': 'svetlo', 'svetla': 'svetlo', 'svetlom': 'svetlo',
            'biológia': 'biológia', 'biológie': 'biológia', 'biológii': 'biológia',
            'kvantová': 'kvantová biológia', 'kvantovej': 'kvantová biológia',
            'zdravie': 'zdravie', 'zdravia': 'zdravie', 'zdravím': 'zdravie',
            'vplyv': 'vplyv', 'vplyvom': 'vplyv', 'vplyvu': 'vplyv'
        }
        
        words = query.lower().split()
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in slovak_mappings:
                stems.append(slovak_mappings[clean_word])
        
        return stems
    
    def get_entity_context(self, entity_name: str, max_related: int = 10) -> Optional[EntityContext]:
        """Get comprehensive context about an entity from the knowledge graph."""
        # Check cache first
        cache_key = f"{entity_name}_{max_related}"
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        try:
            with self.neo4j_driver.session() as session:
                # Get entity information and its strongest connections
                query = """
                MATCH (e:Entity)
                WHERE e.name = $entity_name OR e.normalized_name = $entity_name
                WITH e
                OPTIONAL MATCH (e)-[r:CO_OCCURS]-(related:Entity)
                WITH e, related, MAX(r.strength) as strength
                ORDER BY strength DESC
                LIMIT $max_related
                RETURN e.name as entity_name, 
                       e.type as entity_type,
                       e.mention_count as mention_count,
                       collect(DISTINCT {name: related.name, strength: strength}) as related_entities
                """
                
                result = session.run(query, 
                                   entity_name=entity_name, 
                                   max_related=max_related).single()
                
                if not result:
                    return None
                
                entity_context = EntityContext(
                    entity_name=result["entity_name"],
                    entity_type=result["entity_type"],
                    mention_count=result["mention_count"] or 0,
                    related_entities=[(rel["name"], rel["strength"]) 
                                    for rel in result["related_entities"]]
                )
                
                # Cache the result
                self.entity_cache[cache_key] = entity_context
                return entity_context
                
        except Exception as e:
            print(f"⚠️  Graph context error for '{entity_name}': {e}")
            return None
    
    def get_entity_neighborhood(self, entity_names: List[str], 
                               max_hops: int = 2, 
                               max_nodes: int = 50) -> Set[str]:
        """Get the neighborhood of entities in the knowledge graph."""
        if not entity_names:
            return set()
        
        try:
            with self.neo4j_driver.session() as session:
                # Multi-hop traversal to find related entities
                query = """
                MATCH (start:Entity)-[r:CO_OCCURS*1..{max_hops}]-(neighbor:Entity)
                WHERE start.name IN $entity_names 
                   OR start.normalized_name IN $entity_names
                WITH neighbor, sum(r[0].strength) as total_strength
                ORDER BY total_strength DESC
                LIMIT $max_nodes
                RETURN collect(neighbor.name) as neighborhood
                """.format(max_hops=max_hops)
                
                result = session.run(query, 
                                   entity_names=entity_names,
                                   max_nodes=max_nodes).single()
                
                if result and result["neighborhood"]:
                    return set(result["neighborhood"])
                return set()
                
        except Exception as e:
            print(f"⚠️  Neighborhood retrieval error: {e}")
            return set()
    
    def vector_search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            # Prepare query for embedding (same as in embedding_generator.py)
            prepared_query = f"query: {query.strip()}"
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([prepared_query], normalize_embeddings=True)[0]
            
            # Search in ChromaDB
            results = self.chroma_collection_obj.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to consistent format
            vector_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                vector_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': similarity,
                    'rank': i + 1
                })
            
            return vector_results
            
        except Exception as e:
            print(f"⚠️  Vector search error: {e}")
            return []
    
    def graph_enhanced_search(self, query: str, 
                            vector_results: List[Dict[str, Any]],
                            query_entities: List[str]) -> List[GraphRAGResult]:
        """Enhance vector search results with graph-based context."""
        enhanced_results = []
        
        # Get entity contexts for query entities
        entity_contexts = {}
        for entity in query_entities:
            context = self.get_entity_context(entity)
            if context:
                entity_contexts[entity] = context
        
        # Get neighborhood of query entities
        entity_neighborhood = self.get_entity_neighborhood(query_entities)
        
        for vector_result in vector_results:
            try:
                # Extract entities from the retrieved content
                content_entities = self.extract_query_entities(vector_result['content'])
                
                # Calculate graph relevance score
                graph_relevance = self._calculate_graph_relevance(
                    content_entities, query_entities, entity_contexts, entity_neighborhood
                )
                
                # Find entity relationships mentioned in this content
                entity_relationships = self._find_entity_relationships(
                    content_entities, entity_contexts
                )
                
                # Calculate combined score
                vector_score = vector_result['similarity_score']
                combined_score = self._combine_scores(vector_score, graph_relevance)
                
                # Create enhanced result
                enhanced_result = GraphRAGResult(
                    content=vector_result['content'],
                    source_url=vector_result['metadata'].get('source_url', ''),
                    source_title=vector_result['metadata'].get('source_title', ''),
                    similarity_score=vector_score,
                    graph_relevance_score=graph_relevance,
                    combined_score=combined_score,
                    related_entities=content_entities,
                    entity_relationships=entity_relationships,
                    chunk_metadata=vector_result['metadata']
                )
                
                enhanced_results.append(enhanced_result)
                
            except Exception as e:
                print(f"⚠️  Error enhancing result: {e}")
                continue
        
        # Sort by combined score
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        return enhanced_results
    
    def _calculate_graph_relevance(self, content_entities: List[str], 
                                 query_entities: List[str],
                                 entity_contexts: Dict[str, EntityContext],
                                 entity_neighborhood: Set[str]) -> float:
        """Calculate how relevant content is based on graph structure."""
        if not content_entities or not query_entities:
            return 0.0
        
        relevance_score = 0.0
        total_possible = len(query_entities)
        
        for query_entity in query_entities:
            entity_score = 0.0
            
            # Direct entity match
            if query_entity in content_entities:
                entity_score += 1.0
            
            # Related entity match (entities connected to query entity)
            if query_entity in entity_contexts:
                context = entity_contexts[query_entity]
                related_names = [name.lower() for name, _ in context.related_entities]
                
                for content_entity in content_entities:
                    if content_entity in related_names:
                        # Score based on connection strength
                        for name, strength in context.related_entities:
                            if name.lower() == content_entity:
                                # Normalize strength (assume max strength is around 1000)
                                entity_score += min(strength / 1000.0, 0.5)
                                break
            
            # Neighborhood match
            neighborhood_matches = sum(1 for entity in content_entities 
                                     if entity in entity_neighborhood)
            if neighborhood_matches > 0:
                entity_score += 0.3 * min(neighborhood_matches / len(content_entities), 1.0)
            
            relevance_score += min(entity_score, 1.0)
        
        return relevance_score / total_possible if total_possible > 0 else 0.0
    
    def _find_entity_relationships(self, content_entities: List[str],
                                 entity_contexts: Dict[str, EntityContext]) -> List[Dict[str, Any]]:
        """Find relationships between entities mentioned in the content."""
        relationships = []
        
        for entity1 in content_entities:
            for entity_name, context in entity_contexts.items():
                if entity_name == entity1:
                    for related_name, strength in context.related_entities[:5]:  # Top 5
                        if related_name.lower() in content_entities:
                            relationships.append({
                                'entity1': entity1,
                                'entity2': related_name.lower(),
                                'strength': strength,
                                'relationship_type': 'CO_OCCURS'
                            })
        
        return relationships
    
    def _combine_scores(self, vector_score: float, graph_relevance: float,
                       vector_weight: float = 0.7, graph_weight: float = 0.3) -> float:
        """Combine vector similarity and graph relevance scores with dynamic weighting."""
        # Dynamic weighting based on graph relevance
        # If graph relevance is high, increase its weight
        if graph_relevance > 0.8:
            graph_weight = min(0.5, graph_weight + 0.1)
            vector_weight = 1.0 - graph_weight
        elif graph_relevance < 0.2:
            # If graph relevance is low, rely more on vector similarity
            vector_weight = min(0.9, vector_weight + 0.1)
            graph_weight = 1.0 - vector_weight
        
        return (vector_score * vector_weight) + (graph_relevance * graph_weight)
    
    def search(self, query: str, 
               n_results: int = 10,
               vector_weight: float = 0.7,
               graph_weight: float = 0.3,
               verbose: bool = True) -> List[GraphRAGResult]:
        """
        Perform GraphRAG search combining vector similarity and graph traversal.
        
        Args:
            query: User query
            n_results: Number of results to return
            vector_weight: Weight for vector similarity (0.0-1.0)
            graph_weight: Weight for graph relevance (0.0-1.0)
            verbose: Whether to print search progress
            
        Returns:
            List of GraphRAGResult objects ranked by combined score
        """
        if verbose:
            print(f"🔍 GraphRAG Search: '{query}'")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract entities from query
            query_entities = self.extract_query_entities(query)
            if verbose:
                print(f"   🏷️  Query entities: {query_entities}")
            
            # Step 2: Vector similarity search
            if verbose:
                print(f"   📊 Performing vector search...")
            vector_results = self.vector_search(query, n_results * 2)  # Get more for reranking
            
            if not vector_results:
                self.logger.warning(f"No vector results found for query: {query}")
                return []
            
            # Step 3: Graph-enhanced reranking
            if verbose:
                print(f"   🌐 Enhancing with graph context...")
            enhanced_results = self.graph_enhanced_search(query, vector_results, query_entities)
            
            # Step 4: Return top results with quality filtering
            final_results = self._filter_and_rank_results(enhanced_results, n_results)
            
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"   ⏱️  Search completed in {elapsed_time:.2f}s")
                print(f"   📈 Returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Search error for query '{query}': {e}")
            if verbose:
                print(f"   ❌ Search failed: {e}")
            return []
    
    def _filter_and_rank_results(self, results: List[GraphRAGResult], n_results: int) -> List[GraphRAGResult]:
        """Filter and rank results based on quality metrics."""
        # Filter out low-quality results
        filtered_results = []
        for result in results:
            # Quality thresholds
            if (result.combined_score > 0.3 and  # Minimum combined score
                result.similarity_score > 0.2 and  # Minimum vector similarity
                len(result.content.strip()) > 50):  # Minimum content length
                filtered_results.append(result)
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top n results
        return filtered_results[:n_results]
    
    def get_entity_summary(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get a comprehensive summary of an entity from the knowledge graph."""
        context = self.get_entity_context(entity_name, max_related=20)
        if not context:
            return None
        
        # Get chunks that mention this entity
        try:
            # Search for content containing this entity
            entity_results = self.vector_search(entity_name, n_results=5)
            
            return {
                'entity_name': context.entity_name,
                'entity_type': context.entity_type,
                'mention_count': context.mention_count,
                'top_related_entities': context.related_entities[:10],
                'sample_content': [r['content'][:200] + '...' for r in entity_results[:3]],
                'source_articles': list(set([
                    r['metadata'].get('source_title', 'Unknown') 
                    for r in entity_results
                ]))
            }
        except Exception as e:
            print(f"⚠️  Error getting entity summary: {e}")
            return None
    
    def close(self):
        """Close database connections."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        print("🔌 GraphRAG system connections closed")


def main():
    """Demo GraphRAG system."""
    print("Slovak Health GraphRAG System")
    print("=" * 50)
    
    try:
        # Initialize GraphRAG system
        graphrag = SlovakHealthGraphRAG()
        
        # Demo queries
        demo_queries = [
            "Ako mitochondrie súvisia s ATP a energiou?",
            "Čo je kvantová biológia a svetlo?",
            "Ako DHA ovplyvňuje zdravie mitochondrií?",
            "Aký je vplyv chladného šoku na hormóny?"
        ]
        
        print(f"\n🧪 Testing GraphRAG with demo queries:")
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*60}")
            print(f"Demo Query {i}: {query}")
            print('='*60)
            
            results = graphrag.search(query, n_results=3)
            
            for j, result in enumerate(results, 1):
                print(f"\n📋 Result {j} (Combined Score: {result.combined_score:.3f})")
                print(f"   📊 Vector: {result.similarity_score:.3f} | Graph: {result.graph_relevance_score:.3f}")
                print(f"   📄 Source: {result.source_title}")
                print(f"   🏷️  Entities: {', '.join(result.related_entities[:5])}")
                print(f"   📝 Content: {result.content[:150]}...")
                
                if result.entity_relationships:
                    print(f"   🔗 Relationships: {len(result.entity_relationships)} found")
        
        # Demo entity summary
        print(f"\n{'='*60}")
        print("Entity Summary Demo")
        print('='*60)
        
        entity_summary = graphrag.get_entity_summary("mitochondria")
        if entity_summary:
            print(f"📊 Entity: {entity_summary['entity_name']}")
            print(f"   Type: {entity_summary['entity_type']}")
            print(f"   Mentions: {entity_summary['mention_count']}")
            print(f"   Top related: {[name for name, _ in entity_summary['top_related_entities'][:5]]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'graphrag' in locals():
            graphrag.close()


if __name__ == "__main__":
    main()