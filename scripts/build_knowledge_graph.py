#!/usr/bin/env python3

"""
Versioned Knowledge Graph Builder for Multi-Source Slovak Health Content.

This script creates versioned knowledge graphs from different source types
(blog articles, forum posts, books, videos, scientific papers) with complete
provenance tracking and the ability to switch between versions.

Usage:
    # Create KB from cleaned blog articles
    python scripts/build_knowledge_graph.py --source-type blog --input-dir data/raw/scraped_data/articles/cleaned_articles/articles/
    
    # Create KB with multiple sources
    python scripts/build_knowledge_graph.py --source-types blog,forum --version-name "blog_forum_combined"
    
    # List all versions
    python scripts/build_knowledge_graph.py --list-versions
    
    # Activate a version for GraphRAG
    python scripts/build_knowledge_graph.py --activate v1_20250117_blog_only
"""

import argparse
import json
import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VersionedKnowledgeGraphBuilder:
    """Builds and manages versioned knowledge graphs with complete provenance tracking."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.kg_base_dir = self.project_root / "data" / "knowledge_graphs"
        self.kg_base_dir.mkdir(exist_ok=True, parents=True)
        
        # Source type configurations
        self.source_configs = {
            "blog": {
                "default_path": "data/raw/scraped_data/articles/cleaned_articles/articles/",
                "description": "Blog articles from jaroslavlachky.sk",
                "file_pattern": "*.json"
            },
            "forum": {
                "default_path": "data/raw/scraped_data/forum/",
                "description": "Forum discussions from jaroslavlachky.sk/forum/",
                "file_pattern": "*.json"
            },
            "books": {
                "default_path": "data/raw/scraped_data/books/",
                "description": "PDF books with images",
                "file_pattern": "*.json"
            },
            "videos": {
                "default_path": "data/raw/scraped_data/videos/",
                "description": "Video and podcast transcripts",
                "file_pattern": "*.json"
            },
            "papers": {
                "default_path": "data/raw/scraped_data/pdfs/",
                "description": "Scientific papers and research",
                "file_pattern": "*.json"
            }
        }
    
    def generate_version_name(self, source_types: List[str], custom_name: Optional[str] = None) -> str:
        """Generate a version name based on date and source types."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_name:
            return f"v_{date_str}_{custom_name}"
        else:
            source_str = "_".join(sorted(source_types))
            return f"v_{date_str}_{source_str}"
    
    def create_version_metadata(self, version_name: str, source_configs: Dict[str, Dict], 
                              description: str = "") -> Dict[str, Any]:
        """Create metadata for a knowledge graph version."""
        metadata = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "sources": {},
            "processing_stats": {},
            "neo4j_database": f"neo4j_{version_name.replace('v_', '').replace('_', '')}",
            "chromadb_collections": []
        }
        
        # Analyze each source
        for source_type, config in source_configs.items():
            source_path = Path(config["input_path"])
            if source_path.exists():
                files = list(source_path.glob(self.source_configs[source_type]["file_pattern"]))
                total_words = 0
                
                # Calculate total words for JSON files
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict) and 'content' in data:
                                total_words += len(data['content'].split())
                            elif isinstance(data, dict) and 'word_count' in data:
                                total_words += data['word_count']
                    except Exception as e:
                        logger.warning(f"Could not process {file_path}: {e}")
                
                metadata["sources"][source_type] = {
                    "path": str(source_path),
                    "description": self.source_configs[source_type]["description"],
                    "file_count": len(files),
                    "total_words": total_words
                }
                
                # ChromaDB collection names
                metadata["chromadb_collections"].append(f"{version_name}_{source_type}_chunks")
        
        return metadata
    
    def create_version_structure(self, version_name: str) -> Path:
        """Create the directory structure for a new version."""
        version_dir = self.kg_base_dir / version_name
        
        if version_dir.exists():
            response = input(f"Version {version_name} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
            shutil.rmtree(version_dir)
        
        # Create directory structure
        version_dir.mkdir(parents=True)
        (version_dir / "chunked_data").mkdir()
        (version_dir / "embeddings").mkdir()
        (version_dir / "neo4j_export").mkdir()
        
        logger.info(f"Created version structure: {version_dir}")
        return version_dir
    
    def run_content_chunker(self, version_dir: Path, source_configs: Dict[str, Dict]) -> bool:
        """Run content chunker with custom input directories."""
        logger.info("🔄 Running content chunker...")
        
        # Prepare input directories
        input_dirs = [config["input_path"] for config in source_configs.values()]
        output_dir = version_dir / "chunked_data"
        
        try:
            # Import and run content chunker
            sys.path.append(str(self.project_root / "scripts" / "data_processing"))
            from content_chunker import ContentChunker
            
            chunker = ContentChunker(
                input_dirs=input_dirs,
                output_dir=str(output_dir)
            )
            
            chunker.process_all_articles()
            
            # Save the chunks to JSON file
            chunker.save_chunks()
            
            logger.info(f"✅ Content chunking completed. Output: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Content chunking failed: {e}")
            return False
    
    def run_entity_extraction(self, version_dir: Path) -> bool:
        """Run entity extraction on chunked content."""
        logger.info("🔍 Running entity extraction...")
        
        chunked_file = version_dir / "chunked_data" / "chunked_content.json"
        if not chunked_file.exists():
            logger.error(f"Chunked content file not found: {chunked_file}")
            return False
        
        try:
            sys.path.append(str(self.project_root / "scripts" / "data_processing"))
            from entity_extractor import SlovakHealthEntityExtractor
            
            extractor = SlovakHealthEntityExtractor()
            entities_data = extractor.process_chunked_content(str(chunked_file))
            
            # Save entities
            output_file = version_dir / "chunked_data" / "extracted_entities.json"
            extractor.save_extracted_entities(entities_data, str(output_file))
            
            logger.info(f"✅ Entity extraction completed. Output: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return False
    
    def run_embedding_generation(self, version_dir: Path, version_name: str) -> bool:
        """Run embedding generation with version-specific collections."""
        logger.info("🤖 Running embedding generation...")
        
        chunked_file = version_dir / "chunked_data" / "chunked_content.json"
        if not chunked_file.exists():
            logger.error(f"Chunked content file not found: {chunked_file}")
            return False
        
        try:
            sys.path.append(str(self.project_root / "scripts" / "data_processing"))
            from embedding_generator import EmbeddingGenerator
            
            # Use version-specific collection name and output directory
            collection_name = f"{version_name}_chunks"
            output_dir = version_dir / "embeddings"
            
            generator = EmbeddingGenerator(
                input_file=str(chunked_file),
                db_path=str(output_dir / "vector_db"),
                collection_name=collection_name
            )
            
            generator.process_all()
            
            logger.info(f"✅ Embedding generation completed. Collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return False
    
    def run_graph_building(self, version_dir: Path, metadata: Dict[str, Any]) -> bool:
        """Run Neo4j graph building with version-specific database."""
        logger.info("🌐 Running Neo4j graph building...")
        
        entities_file = version_dir / "chunked_data" / "extracted_entities.json"
        if not entities_file.exists():
            logger.error(f"Entities file not found: {entities_file}")
            return False
        
        try:
            sys.path.append(str(self.project_root / "scripts" / "data_processing"))
            from neo4j_graph_builder import SlovakHealthGraphBuilder
            
            # Use version-specific database name
            db_name = metadata["neo4j_database"]
            
            builder = SlovakHealthGraphBuilder(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="healthgraph123"
            )
            
            if not builder.connect():
                logger.error("Failed to connect to Neo4j")
                return False
            
            # Build the graph
            processed_data = builder.load_and_process_entities(str(entities_file))
            builder.create_entities(processed_data['entities'], processed_data['entity_chunk_map'])
            builder.create_articles_and_chunks(processed_data['chunk_article_map'], processed_data['raw_chunk_entities'])
            builder.create_relationships(processed_data['entity_chunk_map'])
            builder.create_entity_chunk_relationships(processed_data['raw_chunk_entities'])
            
            # Get statistics
            stats = builder.get_graph_statistics()
            metadata["processing_stats"].update(stats)
            
            # Export graph for backup
            export_file = version_dir / "neo4j_export" / "graph_export.cypher"
            # Note: We could add graph export functionality here if needed
            
            builder.close()
            
            logger.info(f"✅ Graph building completed. Database: {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Graph building failed: {e}")
            return False
    
    def save_metadata(self, version_dir: Path, metadata: Dict[str, Any]):
        """Save version metadata to JSON file."""
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Metadata saved: {metadata_file}")
    
    def update_current_symlink(self, version_name: str):
        """Update the 'current' symlink to point to the new version."""
        current_link = self.kg_base_dir / "current"
        version_dir = self.kg_base_dir / version_name
        
        # Remove existing symlink
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        
        # Create new symlink
        current_link.symlink_to(version_name)
        logger.info(f"🔗 Updated 'current' symlink to point to {version_name}")
    
    def list_versions(self):
        """List all available knowledge graph versions."""
        print("\n📚 Available Knowledge Graph Versions:")
        print("=" * 60)
        
        current_link = self.kg_base_dir / "current"
        current_version = None
        if current_link.exists() and current_link.is_symlink():
            current_version = current_link.readlink().name
        
        versions = []
        for version_dir in self.kg_base_dir.iterdir():
            if version_dir.is_dir() and version_dir.name != "current":
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    versions.append((version_dir.name, metadata))
        
        # Sort by creation date
        versions.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)
        
        for version_name, metadata in versions:
            status = " (CURRENT)" if version_name == current_version else ""
            print(f"\n📦 {version_name}{status}")
            print(f"   Created: {metadata.get('created_at', 'Unknown')}")
            print(f"   Description: {metadata.get('description', 'No description')}")
            print(f"   Sources: {', '.join(metadata.get('sources', {}).keys())}")
            
            # Show source details
            for source_type, source_info in metadata.get('sources', {}).items():
                print(f"     - {source_type}: {source_info.get('file_count', 0)} files, {source_info.get('total_words', 0):,} words")
            
            # Show processing stats if available
            stats = metadata.get('processing_stats', {})
            if stats:
                print(f"   Stats: {stats.get('entities_created', 0)} entities, {stats.get('relationships_created', 0)} relationships")
    
    def activate_version(self, version_name: str):
        """Activate a specific version for GraphRAG."""
        version_dir = self.kg_base_dir / version_name
        if not version_dir.exists():
            logger.error(f"Version {version_name} does not exist")
            return False
        
        self.update_current_symlink(version_name)
        logger.info(f"✅ Activated version: {version_name}")
        return True
    
    def build_knowledge_graph(self, source_types: List[str], source_paths: Dict[str, str],
                            version_name: Optional[str] = None, description: str = "") -> bool:
        """Build a complete knowledge graph from specified sources."""
        
        # Generate version name
        if not version_name:
            version_name = self.generate_version_name(source_types)
        else:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"v_{date_str}_{version_name}"
        
        logger.info(f"🚀 Building knowledge graph version: {version_name}")
        
        # Prepare source configurations
        source_configs = {}
        for source_type in source_types:
            if source_type not in self.source_configs:
                logger.error(f"Unknown source type: {source_type}")
                return False
            
            # Use custom path or default
            input_path = source_paths.get(source_type, self.source_configs[source_type]["default_path"])
            input_path = self.project_root / input_path
            
            if not input_path.exists():
                logger.error(f"Source path does not exist: {input_path}")
                return False
            
            source_configs[source_type] = {
                "input_path": str(input_path),
                "description": self.source_configs[source_type]["description"]
            }
        
        # Create version structure
        version_dir = self.create_version_structure(version_name)
        
        # Create metadata
        metadata = self.create_version_metadata(version_name, source_configs, description)
        
        # Run processing pipeline
        success = True
        
        # 1. Content chunking
        if success:
            success = self.run_content_chunker(version_dir, source_configs)
        
        # 2. Entity extraction
        if success:
            success = self.run_entity_extraction(version_dir)
        
        # 3. Embedding generation
        if success:
            success = self.run_embedding_generation(version_dir, version_name)
        
        # 4. Graph building
        if success:
            success = self.run_graph_building(version_dir, metadata)
        
        if success:
            # Save metadata
            self.save_metadata(version_dir, metadata)
            
            # Update current symlink
            self.update_current_symlink(version_name)
            
            logger.info(f"🎉 Successfully created knowledge graph version: {version_name}")
            return True
        else:
            logger.error(f"❌ Failed to create knowledge graph version: {version_name}")
            logger.info(f"📁 Partial version preserved for debugging: {version_dir}")
            # Don't clean up for debugging
            # if version_dir.exists():
            #     shutil.rmtree(version_dir)
            return False


def main():
    parser = argparse.ArgumentParser(description='Versioned Knowledge Graph Builder')
    
    # Main commands
    parser.add_argument('--source-type', type=str, 
                       help='Single source type (blog, forum, books, videos, papers)')
    parser.add_argument('--source-types', type=str,
                       help='Comma-separated source types (e.g., "blog,forum")')
    parser.add_argument('--input-dir', type=str,
                       help='Custom input directory for single source type')
    parser.add_argument('--version-name', type=str,
                       help='Custom version name (will be prefixed with date)')
    parser.add_argument('--description', type=str, default='',
                       help='Description for this knowledge graph version')
    
    # Management commands
    parser.add_argument('--list-versions', action='store_true',
                       help='List all available knowledge graph versions')
    parser.add_argument('--activate', type=str,
                       help='Activate a specific version for GraphRAG')
    
    args = parser.parse_args()
    
    builder = VersionedKnowledgeGraphBuilder()
    
    # Handle management commands
    if args.list_versions:
        builder.list_versions()
        return
    
    if args.activate:
        builder.activate_version(args.activate)
        return
    
    # Handle build commands
    source_types = []
    source_paths = {}
    
    if args.source_type:
        source_types = [args.source_type]
        if args.input_dir:
            source_paths[args.source_type] = args.input_dir
    elif args.source_types:
        source_types = [s.strip() for s in args.source_types.split(',')]
    else:
        parser.print_help()
        return
    
    if not source_types:
        print("❌ No source types specified")
        parser.print_help()
        return
    
    # Build knowledge graph
    success = builder.build_knowledge_graph(
        source_types=source_types,
        source_paths=source_paths,
        version_name=args.version_name,
        description=args.description
    )
    
    if success:
        print(f"\n🎉 Knowledge graph built successfully!")
        print(f"📊 Use --list-versions to see all versions")
        print(f"🔄 Use --activate <version> to switch versions")
    else:
        print(f"\n❌ Knowledge graph build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()