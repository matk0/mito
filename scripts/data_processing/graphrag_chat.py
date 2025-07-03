#!/usr/bin/env python3

"""
Interactive GraphRAG Chat Interface for Slovak Health Knowledge
Simple command-line interface for querying the GraphRAG system.
"""

import sys
from pathlib import Path
import json
from typing import List

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

class GraphRAGChat:
    """Interactive chat interface for GraphRAG system."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.graphrag = None
        print("🚀 Initializing Slovak Health GraphRAG Chat...")
        self._setup_graphrag()
    
    def _setup_graphrag(self):
        """Setup the GraphRAG system."""
        try:
            from graphrag_system import SlovakHealthGraphRAG
            self.graphrag = SlovakHealthGraphRAG()
            print("✅ GraphRAG system ready!")
        except Exception as e:
            print(f"❌ Failed to initialize GraphRAG: {e}")
            print("   Make sure Neo4j is running and ChromaDB is available")
            sys.exit(1)
    
    def format_results(self, results: List, show_details: bool = False):
        """Format search results for display."""
        if not results:
            print("🤷 No results found.")
            return
        
        print(f"\n📊 Found {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            # Header with scores
            print(f"\n📋 Result {i} | Combined Score: {result.combined_score:.3f}")
            print(f"   📊 Vector: {result.similarity_score:.3f} | Graph: {result.graph_relevance_score:.3f}")
            
            # Source information
            print(f"   📄 Source: {result.source_title}")
            if result.source_url:
                print(f"   🔗 URL: {result.source_url}")
            
            # Content preview
            print(f"   📝 Content:")
            content_lines = result.content.split('\n')
            for line in content_lines[:3]:  # Show first 3 lines
                if line.strip():
                    print(f"      {line.strip()}")
            if len(content_lines) > 3:
                print("      ...")
            
            # Entity information
            if result.related_entities:
                entities_str = ", ".join(result.related_entities[:8])
                if len(result.related_entities) > 8:
                    entities_str += f" (+{len(result.related_entities) - 8} more)"
                print(f"   🏷️  Entities: {entities_str}")
            
            # Relationship information
            if result.entity_relationships and show_details:
                print(f"   🔗 Relationships:")
                for rel in result.entity_relationships[:3]:
                    print(f"      {rel['entity1']} ↔ {rel['entity2']} (strength: {rel['strength']})")
            
            print("-" * 60)
    
    def search_command(self, query: str, n_results: int = 5, show_details: bool = False):
        """Execute a search command."""
        if not query.strip():
            print("❌ Please provide a search query.")
            return
        
        try:
            results = self.graphrag.search(query, n_results=n_results)
            self.format_results(results, show_details)
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    def entity_command(self, entity_name: str):
        """Get information about a specific entity."""
        if not entity_name.strip():
            print("❌ Please provide an entity name.")
            return
        
        try:
            summary = self.graphrag.get_entity_summary(entity_name)
            if summary:
                print(f"\n📊 Entity Summary: {summary['entity_name']}")
                print("=" * 60)
                print(f"Type: {summary['entity_type']}")
                print(f"Mentions: {summary['mention_count']}")
                print(f"\nTop Related Entities:")
                for name, strength in summary['top_related_entities'][:10]:
                    print(f"  • {name} (strength: {strength})")
                
                print(f"\nFound in Articles:")
                for article in summary['source_articles'][:5]:
                    print(f"  • {article}")
            else:
                print(f"❌ Entity '{entity_name}' not found in knowledge graph.")
        except Exception as e:
            print(f"❌ Entity lookup error: {e}")
    
    def help_command(self):
        """Show help information."""
        print("\n🤖 Slovak Health GraphRAG Chat Commands:")
        print("=" * 60)
        print("📝 SEARCH COMMANDS:")
        print("   search <query>              - Search for health information")
        print("   search+ <query>             - Search with detailed entity relationships")
        print("   results <n> <query>         - Search with custom number of results")
        print("")
        print("🔍 ENTITY COMMANDS:")
        print("   entity <name>               - Get detailed info about an entity")
        print("   info <name>                 - Alias for entity command")
        print("")
        print("💡 EXAMPLE QUERIES:")
        print("   search mitochondrie a energia")
        print("   search+ kvantová biológia a svetlo")
        print("   results 10 DHA a zdravie")
        print("   entity mitochondria")
        print("")
        print("🎯 OTHER COMMANDS:")
        print("   help                        - Show this help")
        print("   quit, exit                  - Exit the chat")
        print("=" * 60)
    
    def run(self):
        """Run the interactive chat interface."""
        print("\n🎯 Slovak Health GraphRAG Chat Interface")
        print("Type 'help' for commands or start asking health questions!")
        print("Example: search mitochondrie a ATP")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 GraphRAG> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command in ['quit', 'exit']:
                    break
                elif command == 'help':
                    self.help_command()
                elif command == 'search':
                    self.search_command(args)
                elif command == 'search+':
                    self.search_command(args, show_details=True)
                elif command == 'results':
                    # Parse "results N query"
                    result_parts = args.split(' ', 1)
                    if len(result_parts) == 2 and result_parts[0].isdigit():
                        n_results = int(result_parts[0])
                        query = result_parts[1]
                        self.search_command(query, n_results=n_results)
                    else:
                        print("❌ Usage: results <number> <query>")
                elif command in ['entity', 'info']:
                    self.entity_command(args)
                else:
                    # Treat as search query
                    self.search_command(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Thanks for using Slovak Health GraphRAG!")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.graphrag:
            self.graphrag.close()


def main():
    """Main entry point."""
    chat = GraphRAGChat()
    chat.run()


if __name__ == "__main__":
    main()