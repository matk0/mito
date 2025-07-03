#!/usr/bin/env python3

"""
Neo4j Knowledge Graph Builder for Slovak Health Content.
Builds a comprehensive knowledge graph from extracted entities and relationships.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import re

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlovakHealthGraphBuilder:
    """Builds and manages the Slovak health knowledge graph in Neo4j."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "healthgraph123"):
        """
        Initialize the graph builder.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        # Graph statistics
        self.stats = {
            'entities_created': 0,
            'relationships_created': 0,
            'articles_processed': 0,
            'chunks_processed': 0
        }
        
    def connect(self) -> bool:
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    print(f"✅ Connected to Neo4j at {self.uri}")
                    return True
                    
        except ServiceUnavailable:
            print(f"❌ Could not connect to Neo4j at {self.uri}")
            print("   Make sure Neo4j is running and accessible")
            return False
        except AuthError:
            print(f"❌ Authentication failed for user '{self.username}'")
            print("   Check your username and password")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")
    
    def clear_database(self) -> bool:
        """Clear all nodes and relationships from the database."""
        try:
            with self.driver.session() as session:
                print("🧹 Clearing existing data...")
                
                # Delete all relationships first
                session.run("MATCH ()-[r]-() DELETE r")
                
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                
                # Drop all constraints and indexes
                constraints = session.run("SHOW CONSTRAINTS").data()
                for constraint in constraints:
                    constraint_name = constraint.get('name', '')
                    if constraint_name:
                        session.run(f"DROP CONSTRAINT `{constraint_name}` IF EXISTS")
                
                indexes = session.run("SHOW INDEXES").data()
                for index in indexes:
                    index_name = index.get('name', '')
                    if index_name and index.get('type') != 'LOOKUP':  # Don't drop system indexes
                        session.run(f"DROP INDEX `{index_name}` IF EXISTS")
                
                print("✅ Database cleared successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return False
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            print("🏗️  Creating constraints and indexes...")
            
            constraints_and_indexes = [
                # Unique constraints for entity names
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT article_url_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                
                # Indexes for faster lookups
                "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX entity_normalized_idx IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
                "CREATE INDEX article_date_idx IF NOT EXISTS FOR (a:Article) ON (a.date)",
                "CREATE INDEX chunk_position_idx IF NOT EXISTS FOR (c:Chunk) ON (c.position)",
                
                # Full-text search indexes
                "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.normalized_name]",
                "CREATE FULLTEXT INDEX article_search IF NOT EXISTS FOR (a:Article) ON EACH [a.title, a.content_preview]"
            ]
            
            for statement in constraints_and_indexes:
                try:
                    session.run(statement)
                except Exception as e:
                    print(f"⚠️  Could not create constraint/index: {e}")
            
            print("✅ Constraints and indexes created")
    
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for better matching."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', name.strip().lower())
        
        # Remove common Slovak prefixes/suffixes for better matching
        normalized = re.sub(r'^(a|the|na|v|o|do|pre|pri|od|za)\s+', '', normalized)
        
        return normalized
    
    def _is_common_word(self, text: str) -> bool:
        """Check if text is a common Slovak word that should be filtered out."""
        common_slovak_words = {
            # Prepositions
            'v', 'na', 'do', 'pre', 'pri', 'od', 'za', 'so', 'cez', 'po', 'bez', 'okrem',
            'k', 'ku', 'vo', 'nad', 'pod', 'medzi', 'proti', 'podľa', 'kvôli',
            
            # Articles and pronouns
            'a', 'aj', 'ale', 'alebo', 'čo', 'že', 'ako', 'keď', 'ak', 'kto', 'kde',
            'ten', 'tá', 'to', 'tie', 'tých', 'tým', 'tej', 'tom', 'tu', 'tam',
            'jeden', 'jedna', 'jedno', 'jeho', 'jej', 'ich', 'moja', 'môj', 'moje',
            
            # Common verbs
            'je', 'sú', 'bol', 'bola', 'bolo', 'boli', 'má', 'mať', 'môže', 'môžu',
            'bude', 'budú', 'bol', 'bude', 'mal', 'mala', 'mali', 'ide', 'idú',
            
            # Numbers and measurements (single chars/short)
            'k', 'g', 'm', 'l', 's', 'h', 'cm', 'mm', 'kg', 'mg',
            
            # Common adjectives
            'veľký', 'malý', 'dobrý', 'zlý', 'nový', 'starý', 'prvý', 'posledný',
            'celý', 'celá', 'celé', 'každý', 'každá', 'každé', 'iný', 'iná', 'iné',
            
            # Other common words
            'sa', 'si', 'nie', 'áno', 'už', 'ešte', 'len', 'iba', 'tiež', 'takže',
            'teda', 'však', 'pritom', 'navyše', 'okrem', 'hlavne', 'práve',
            'možno', 'určite', 'asi', 'pravdepodobne', 'skutočne', 'vlastne',
            
            # English common words (mixed in Slovak text)
            'and', 'or', 'the', 'of', 'in', 'to', 'for', 'with', 'by', 'from',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            
            # Marketing/promotional terms
            'zdarma', 'premium', 'rok', 'zľava', 'zľavy', 'akcia', 'cenník',
            'predplatné', 'členstvo', 'registrácia', 'newsletter', 'blog'
        }
        
        return text.lower().strip() in common_slovak_words
    
    def _is_low_quality_entity(self, entity: Dict) -> bool:
        """Check if entity is low quality and should be filtered."""
        text = entity['text'].strip()
        
        # Filter very short entities (1-2 characters) except known abbreviations
        if len(text) <= 2:
            known_abbreviations = {
                'UV', 'IR', 'LED', 'ATP', 'DNA', 'RNA', 'DHA', 'EPA', 'NAD', 'FAD',
                'mV', 'nm', 'Hz', 'pH', 'O2', 'H2', 'CO2', 'NO', 'Ca', 'Mg', 'Fe',
                'Zn', 'Cu', 'Se', 'I', 'K', 'Na', 'Cl', 'P', 'S', 'N', 'C'
            }
            if text.upper() not in known_abbreviations:
                return True
        
        # Filter entities that are just numbers
        if re.match(r'^\d+$', text):
            return True
        
        # Filter entities that are just punctuation
        if re.match(r'^[^\w\s]+$', text):
            return True
        
        # Filter entities with very low confidence
        if entity.get('confidence', 1.0) < 0.3:
            return True
        
        # Filter measurement units that are too generic
        if entity.get('label') == 'MEASUREMENT_UNIT' and text.lower() in ['v', 'k', 'm', 'l', 's', 'h']:
            return True
        
        # Filter chemical formulas that are too generic
        if entity.get('label') == 'CHEMICAL_FORMULA' and len(text) <= 3:
            generic_chemicals = {'sa', 'na', 'ak', 'no', 'to', 'je', 'ma', 'tu'}
            if text.lower() in generic_chemicals:
                return True
        
        return False
    
    def merge_similar_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge entities that refer to the same concept."""
        # Group by normalized name
        entity_groups = defaultdict(list)
        
        for entity in entities:
            normalized = self.normalize_entity_name(entity['text'])
            entity_groups[normalized].append(entity)
        
        merged_entities = []
        
        for normalized_name, group in entity_groups.items():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Merge multiple entities into one
                main_entity = group[0].copy()
                
                # Use the most common text form
                text_counts = Counter(entity['text'] for entity in group)
                main_entity['text'] = text_counts.most_common(1)[0][0]
                
                # Combine all variants
                all_variants = set()
                for entity in group:
                    all_variants.add(entity['text'])
                    if 'variants' in entity:
                        all_variants.update(entity['variants'])
                
                main_entity['variants'] = list(all_variants)
                main_entity['normalized_name'] = normalized_name
                main_entity['mention_count'] = len(group)
                
                merged_entities.append(main_entity)
        
        return merged_entities
    
    def load_and_process_entities(self, entities_file: str) -> Dict[str, Any]:
        """Load and process extracted entities."""
        print(f"📂 Loading entities from {entities_file}")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunk_entities = data['chunk_entities']
        
        print(f"📊 Processing {len(chunk_entities)} chunks")
        
        # Collect all entities
        all_entities = []
        entity_chunk_map = defaultdict(set)
        chunk_article_map = {}
        
        for chunk_data in chunk_entities:
            chunk_id = chunk_data['chunk_id']
            chunk_article_map[chunk_id] = {
                'title': chunk_data['source_title'],
                'url': chunk_data['source_url'],
                'date': chunk_data['source_date'],
                'position': chunk_data['chunk_position']
            }
            
            for entity in chunk_data['entities']:
                # Filter out low-quality entities
                if (entity['label'].startswith('SPACY_') or 
                    entity['label'] == 'CHEMICAL_FORMULA' and len(entity['text']) <= 2 or
                    self._is_common_word(entity['text']) or
                    self._is_low_quality_entity(entity)):
                    continue
                
                all_entities.append(entity)
                entity_text = entity['text'].lower().strip()
                entity_chunk_map[entity_text].add(chunk_id)
        
        print(f"📋 Found {len(all_entities)} entities before merging")
        
        # Merge similar entities
        merged_entities = self.merge_similar_entities(all_entities)
        
        print(f"🔄 Merged to {len(merged_entities)} unique entities")
        
        return {
            'entities': merged_entities,
            'entity_chunk_map': entity_chunk_map,
            'chunk_article_map': chunk_article_map,
            'raw_chunk_entities': chunk_entities
        }
    
    def create_entities(self, entities: List[Dict], entity_chunk_map: Dict[str, Set[int]]):
        """Create entity nodes in Neo4j."""
        print("🏗️  Creating entity nodes...")
        
        with self.driver.session() as session:
            batch_size = 100
            
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                
                # Prepare batch data
                entity_params = []
                for entity in batch:
                    normalized_name = self.normalize_entity_name(entity['text'])
                    mention_count = len(entity_chunk_map.get(entity['text'].lower().strip(), []))
                    
                    entity_params.append({
                        'name': entity['text'],
                        'normalized_name': normalized_name,
                        'type': entity['label'],
                        'confidence': entity.get('confidence', 1.0),
                        'variants': entity.get('variants', []),
                        'mention_count': mention_count
                    })
                
                # Batch create entities
                query = """
                UNWIND $entities AS entity
                CREATE (e:Entity {
                    name: entity.name,
                    normalized_name: entity.normalized_name,
                    type: entity.type,
                    confidence: entity.confidence,
                    variants: entity.variants,
                    mention_count: entity.mention_count,
                    created_at: datetime()
                })
                """
                
                session.run(query, entities=entity_params)
                self.stats['entities_created'] += len(batch)
                
                if (i + batch_size) % 500 == 0:
                    print(f"   📝 Created {i + batch_size} entities...")
        
        print(f"✅ Created {self.stats['entities_created']} entity nodes")
    
    def create_articles_and_chunks(self, chunk_article_map: Dict[int, Dict[str, str]], 
                                   raw_chunk_entities: List[Dict]):
        """Create article and chunk nodes."""
        print("📚 Creating article and chunk nodes...")
        
        # Group chunks by article
        articles = defaultdict(list)
        for chunk_id, article_info in chunk_article_map.items():
            article_key = (article_info['title'], article_info['url'])
            articles[article_key].append(chunk_id)
        
        with self.driver.session() as session:
            # Create article nodes
            for (title, url), chunk_ids in articles.items():
                article_info = chunk_article_map[chunk_ids[0]]  # Get info from first chunk
                
                session.run("""
                CREATE (a:Article {
                    title: $title,
                    url: $url,
                    date: $date,
                    chunk_count: $chunk_count,
                    created_at: datetime()
                })
                """, title=title, url=url, date=article_info['date'], chunk_count=len(chunk_ids))
                
                self.stats['articles_processed'] += 1
            
            # Create chunk nodes and link to articles
            for chunk_data in raw_chunk_entities:
                session.run("""
                MATCH (a:Article {url: $url})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    position: $position,
                    entity_count: $entity_count,
                    created_at: datetime()
                })
                CREATE (c)-[:BELONGS_TO]->(a)
                """, 
                url=chunk_data['source_url'],
                chunk_id=chunk_data['chunk_id'],
                position=chunk_data['chunk_position'],
                entity_count=len(chunk_data['entities']))
                
                self.stats['chunks_processed'] += 1
        
        print(f"✅ Created {self.stats['articles_processed']} articles and {self.stats['chunks_processed']} chunks")
    
    def create_relationships(self, entity_chunk_map: Dict[str, Set[int]]):
        """Create relationships between entities based on co-occurrence."""
        print("🔗 Creating entity relationships...")
        
        # Calculate co-occurrence relationships
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        # Group entities by chunks
        chunk_entities = defaultdict(set)
        for entity_text, chunk_ids in entity_chunk_map.items():
            for chunk_id in chunk_ids:
                chunk_entities[chunk_id].add(entity_text)
        
        # Calculate co-occurrences
        relationship_count = 0
        for chunk_id, entities in chunk_entities.items():
            entities_list = list(entities)
            for i, entity1 in enumerate(entities_list):
                for entity2 in entities_list[i+1:]:
                    if entity1 != entity2:
                        co_occurrence[entity1][entity2] += 1
                        co_occurrence[entity2][entity1] += 1
        
        # Create relationships in Neo4j
        with self.driver.session() as session:
            batch_size = 1000
            relationship_batch = []
            
            for entity1, connections in co_occurrence.items():
                for entity2, strength in connections.items():
                    if strength >= 2:  # Minimum co-occurrence threshold
                        relationship_batch.append({
                            'entity1': entity1,
                            'entity2': entity2,
                            'strength': strength,
                            'type': 'CO_OCCURS'
                        })
                        
                        if len(relationship_batch) >= batch_size:
                            self._create_relationship_batch(session, relationship_batch)
                            relationship_count += len(relationship_batch)
                            relationship_batch = []
            
            # Create remaining relationships
            if relationship_batch:
                self._create_relationship_batch(session, relationship_batch)
                relationship_count += len(relationship_batch)
        
        self.stats['relationships_created'] = relationship_count
        print(f"✅ Created {relationship_count} relationships")
    
    def _create_relationship_batch(self, session, relationships: List[Dict]):
        """Create a batch of relationships."""
        query = """
        UNWIND $relationships AS rel
        MATCH (e1:Entity {name: rel.entity1})
        MATCH (e2:Entity {name: rel.entity2})
        CREATE (e1)-[:CO_OCCURS {
            strength: rel.strength,
            created_at: datetime()
        }]->(e2)
        """
        
        session.run(query, relationships=relationships)
    
    def create_entity_chunk_relationships(self, raw_chunk_entities: List[Dict]):
        """Create relationships between entities and chunks."""
        print("📎 Linking entities to chunks...")
        
        with self.driver.session() as session:
            batch_size = 1000
            link_batch = []
            
            for chunk_data in raw_chunk_entities:
                chunk_id = chunk_data['chunk_id']
                
                for entity in chunk_data['entities']:
                    # Skip low-quality entities
                    if (entity['label'].startswith('SPACY_') or 
                        entity['label'] == 'CHEMICAL_FORMULA' and len(entity['text']) <= 2 or
                        self._is_common_word(entity['text']) or
                        self._is_low_quality_entity(entity)):
                        continue
                    
                    link_batch.append({
                        'chunk_id': chunk_id,
                        'entity_name': entity['text'],
                        'confidence': entity.get('confidence', 1.0)
                    })
                    
                    if len(link_batch) >= batch_size:
                        self._create_entity_chunk_links(session, link_batch)
                        link_batch = []
            
            if link_batch:
                self._create_entity_chunk_links(session, link_batch)
        
        print("✅ Entity-chunk relationships created")
    
    def _create_entity_chunk_links(self, session, links: List[Dict]):
        """Create entity-chunk relationship batch."""
        query = """
        UNWIND $links AS link
        MATCH (e:Entity {name: link.entity_name})
        MATCH (c:Chunk {chunk_id: link.chunk_id})
        CREATE (e)-[:MENTIONED_IN {
            confidence: link.confidence,
            created_at: datetime()
        }]->(c)
        """
        
        session.run(query, links=links)
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        print("📊 Generating graph statistics...")
        
        with self.driver.session() as session:
            stats = {}
            
            # Node counts
            result = session.run("MATCH (n:Entity) RETURN count(n) as entity_count").single()
            stats['entity_count'] = result['entity_count']
            
            result = session.run("MATCH (n:Article) RETURN count(n) as article_count").single()
            stats['article_count'] = result['article_count']
            
            result = session.run("MATCH (n:Chunk) RETURN count(n) as chunk_count").single()
            stats['chunk_count'] = result['chunk_count']
            
            # Relationship counts
            result = session.run("MATCH ()-[r:CO_OCCURS]->() RETURN count(r) as co_occur_count").single()
            stats['co_occurrence_relationships'] = result['co_occur_count']
            
            result = session.run("MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) as mention_count").single()
            stats['mention_relationships'] = result['mention_count']
            
            # Entity type distribution
            entity_types = session.run("""
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """).data()
            stats['entity_types'] = entity_types
            
            # Most connected entities
            top_entities = session.run("""
                MATCH (e:Entity)-[r:CO_OCCURS]-()
                RETURN e.name as entity, e.type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
            """).data()
            stats['most_connected_entities'] = top_entities
            
            return stats
    
    def build_complete_graph(self, entities_file: str = "./chunked_data/extracted_entities.json") -> bool:
        """Build the complete knowledge graph."""
        print("🚀 Building Slovak Health Knowledge Graph")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Load and process data
        processed_data = self.load_and_process_entities(entities_file)
        
        # Step 2: Clear existing data and create constraints
        if not self.clear_database():
            return False
        
        self.create_constraints_and_indexes()
        
        # Step 3: Create nodes
        self.create_entities(processed_data['entities'], processed_data['entity_chunk_map'])
        self.create_articles_and_chunks(processed_data['chunk_article_map'], 
                                       processed_data['raw_chunk_entities'])
        
        # Step 4: Create relationships
        self.create_relationships(processed_data['entity_chunk_map'])
        self.create_entity_chunk_relationships(processed_data['raw_chunk_entities'])
        
        # Step 5: Generate statistics
        graph_stats = self.get_graph_statistics()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ KNOWLEDGE GRAPH CONSTRUCTION COMPLETE!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"📊 Graph Statistics:")
        print(f"   • Entities: {graph_stats['entity_count']}")
        print(f"   • Articles: {graph_stats['article_count']}")
        print(f"   • Chunks: {graph_stats['chunk_count']}")
        print(f"   • Co-occurrence relationships: {graph_stats['co_occurrence_relationships']}")
        print(f"   • Mention relationships: {graph_stats['mention_relationships']}")
        
        print(f"\n🔝 Top Entity Types:")
        for entity_type in graph_stats['entity_types'][:5]:
            print(f"   • {entity_type['type']}: {entity_type['count']}")
        
        print(f"\n🌐 Most Connected Entities:")
        for entity in graph_stats['most_connected_entities'][:5]:
            print(f"   • {entity['entity']} ({entity['type']}): {entity['connections']} connections")
        
        print(f"\n🌐 Access your knowledge graph:")
        print(f"   • Neo4j Browser: http://localhost:7474")
        print(f"   • Username: {self.username}")
        print(f"   • Try this query: MATCH (e:Entity)-[r:CO_OCCURS]-(e2) WHERE e.name = 'mitochondrie' RETURN e, r, e2 LIMIT 20")
        
        return True


def main():
    """Main function to build the knowledge graph."""
    print("Slovak Health Knowledge Graph Builder")
    print("=" * 50)
    
    # Check if entities file exists
    entities_file = "./chunked_data/extracted_entities.json"
    if not Path(entities_file).exists():
        print(f"❌ Entities file not found: {entities_file}")
        print("Please run entity extraction first!")
        return
    
    # Get connection details from user
    print("\n🔧 Neo4j Connection Setup")
    print("Default settings:")
    print("  URI: bolt://localhost:7687")
    print("  Username: neo4j")
    print("  Password: healthgraph123")
    
    use_defaults = input("\nUse default settings? (y/n): ").lower().strip()
    
    if use_defaults == 'y':
        builder = SlovakHealthGraphBuilder()
    else:
        uri = input("Neo4j URI [bolt://localhost:7687]: ").strip() or "bolt://localhost:7687"
        username = input("Username [neo4j]: ").strip() or "neo4j"
        password = input("Password: ").strip()
        
        if not password:
            print("❌ Password is required!")
            return
        
        builder = SlovakHealthGraphBuilder(uri, username, password)
    
    # Connect to database
    if not builder.connect():
        print("\n💡 Setup help:")
        print("   1. Make sure Neo4j is running")
        print("   2. Check the NEO4J_SETUP.md file for installation instructions")
        print("   3. Verify your connection details")
        return
    
    try:
        # Build the graph
        success = builder.build_complete_graph(entities_file)
        
        if success:
            print("\n🎉 Your Slovak Health Knowledge Graph is ready!")
            print("   Open Neo4j Browser and start exploring!")
        else:
            print("\n💥 Graph construction failed!")
            
    finally:
        builder.close()


if __name__ == "__main__":
    main()