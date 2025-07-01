#!/usr/bin/env python3

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import argparse

class SlovakHealthChatbot:
    def __init__(self, 
                 db_path: str = "./vector_db",
                 model_name: str = "intfloat/multilingual-e5-large",
                 collection_name: str = "slovak_blog_chunks"):
        """
        Initialize the Slovak Health Chatbot.
        
        Args:
            db_path: Path to ChromaDB vector database
            model_name: SentenceTransformer model for embeddings
            collection_name: Name of the ChromaDB collection
        """
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.collection_name = collection_name
        
        print("🤖 Initializing Slovak Health Chatbot...")
        print("=" * 50)
        
        # Check if vector database exists
        if not self.db_path.exists():
            print("❌ Vector database not found!")
            print(f"Please run 'python3.12 embedding_generator.py' first to create the database.")
            sys.exit(1)
        
        # Initialize embedding model
        print(f"📥 Loading embedding model: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        
        # Initialize ChromaDB client
        print(f"🗄️ Connecting to vector database...")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.client.get_collection(collection_name)
            chunk_count = self.collection.count()
            print(f"✅ Connected to collection '{collection_name}' with {chunk_count} chunks")
        except Exception as e:
            print(f"❌ Failed to connect to collection '{collection_name}': {e}")
            sys.exit(1)
        
        print("=" * 50)
        print("🎉 Chatbot ready! Type your questions about health and biology.")
        print("💡 Topics: mitochondria, epigenetics, hormones, nutrition, quantum biology")
        print("🔚 Type 'quit', 'exit', or 'bye' to end the conversation")
        print("=" * 50)
    
    def prepare_query_for_embedding(self, query: str) -> str:
        """
        Prepare user query for embedding generation.
        Add query prefix for multilingual-e5 models.
        """
        return f"query: {query.strip()}"
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant content.
        
        Args:
            query: User question
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        print(f"🔍 Searching knowledge base for: '{query}'")
        
        # Prepare query for embedding
        prepared_query = self.prepare_query_for_embedding(query)
        
        # Generate query embedding
        start_time = time.time()
        query_embedding = self.model.encode([prepared_query], normalize_embeddings=True)[0]
        embedding_time = time.time() - start_time
        
        # Search in vector database
        search_start = time.time()
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        search_time = time.time() - search_start
        
        print(f"⚡ Search completed in {embedding_time:.3f}s + {search_time:.3f}s")
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            formatted_results.append({
                'rank': i + 1,
                'content': doc,
                'similarity': similarity,
                'metadata': metadata
            })
        
        return formatted_results
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate a coherent response based on the query and search results.
        Creates a structured, LLM-like response that synthesizes information.
        """
        if not search_results:
            return self._get_no_results_response()
        
        # Get the most relevant result
        top_result = search_results[0]
        
        # If similarity is very low, return a "not found" response
        if top_result['similarity'] < 0.3:
            return self._get_low_confidence_response(query)
        
        # Generate coherent response
        response = self._synthesize_answer(query, search_results)
        
        return response
    
    def _synthesize_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Synthesize a coherent answer from search results.
        Creates a structured response that answers the specific question.
        """
        top_result = search_results[0]
        
        # Extract key information from top results
        main_content = top_result['content']
        main_title = top_result['metadata']['source_title']
        
        # Create a coherent response based on the topic
        response_parts = []
        
        # Add a direct answer introduction
        if "mitochondri" in query.lower():
            response_parts.append("🧬 **Mitochondrie** sú kľúčové organely v našich bunkách:")
        elif "leptin" in query.lower():
            response_parts.append("🧠 **Leptín** je dôležitý hormón, ktorý:")
        elif "chlad" in query.lower():
            response_parts.append("🥶 **Adaptácia na chlad** ovplyvňuje náš organizmus takto:")
        elif "epigenetik" in query.lower():
            response_parts.append("🧬 **Epigenetika** je vedný odbor, ktorý:")
        elif "hormón" in query.lower() or "hormón" in query.lower():
            response_parts.append("⚗️ **Hormóny** sú chemické posly, ktoré:")
        elif "vitamín" in query.lower():
            response_parts.append("💊 **Vitamíny** sú esenciálne látky, ktoré:")
        elif "svetlo" in query.lower():
            response_parts.append("☀️ **Svetlo** má na náš organizmus tieto účinky:")
        elif "výživa" in query.lower() or "strava" in query.lower():
            response_parts.append("🍎 **Výživa a strava** ovplyvňujú naše zdravie takto:")
        else:
            response_parts.append("📖 **Odpoveď na vašu otázku:**")
        
        response_parts.append("")
        
        # Extract and present key points from the content
        key_points = self._extract_key_points(main_content, query)
        if key_points:
            for point in key_points:
                response_parts.append(f"• {point}")
            response_parts.append("")
        else:
            # Fallback: show first part of content directly
            preview = main_content[:400] + "..." if len(main_content) > 400 else main_content
            response_parts.append(f"• {preview}")
            response_parts.append("")
        
        # Add supporting information from additional sources
        if len(search_results) > 1:
            supporting_info = self._get_supporting_info(search_results[1:3])
            if supporting_info:
                response_parts.append("🔗 **Súvisiace informácie:**")
                for info in supporting_info:
                    response_parts.append(f"• {info}")
                response_parts.append("")
        
        # Add practical implications or tips if available
        practical_tips = self._extract_practical_tips(main_content)
        if practical_tips:
            response_parts.append("💡 **Praktické využitie:**")
            for tip in practical_tips:
                response_parts.append(f"• {tip}")
            response_parts.append("")
        
        # Add source information
        response_parts.append("---")
        response_parts.append(f"📚 **Hlavný zdroj**: {main_title}")
        response_parts.append(f"📊 **Relevancia**: {top_result['similarity']:.1%}")
        
        if len(search_results) > 1:
            response_parts.append(f"🔍 **Ďalšie zdroje**: {len(search_results)-1} súvisiacich článkov")
        
        return "\\n".join(response_parts)
    
    def _extract_key_points(self, content: str, query: str) -> List[str]:
        """Extract key points from content relevant to the query."""
        # Split content into sentences more carefully
        import re
        
        # Split on sentence endings but preserve them
        sentences = re.split(r'(?<=[.!?])\s+', content)
        key_points = []
        
        # Look for sentences that might contain key information
        for sentence in sentences[:8]:  # Check more sentences
            sentence = sentence.strip()
            if len(sentence) > 40 and len(sentence) < 300:  # Better length range
                # Skip sentences that seem incomplete or are just references
                if any(skip_word in sentence.lower() for skip_word in ['http', 'www.', 'článok si môžeš', 'registrácia', 'odkazy']):
                    continue
                
                # Clean up the sentence
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                
                # Make sure it starts with capital letter
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                key_points.append(sentence)
                
                if len(key_points) >= 3:  # Stop when we have enough good points
                    break
        
        return key_points
    
    def _get_supporting_info(self, additional_results: List[Dict[str, Any]]) -> List[str]:
        """Extract supporting information from additional search results."""
        supporting_info = []
        
        for result in additional_results:
            if result['similarity'] > 0.4:  # Only high-quality matches
                title = result['metadata']['source_title']
                # Extract a brief insight from the content
                content = result['content'][:150]
                # Clean up and truncate
                if '.' in content:
                    content = content.split('.')[0] + '.'
                supporting_info.append(f"{title}: {content}")
        
        return supporting_info[:2]  # Max 2 supporting items
    
    def _extract_practical_tips(self, content: str) -> List[str]:
        """Extract practical tips or actionable information from content."""
        tips = []
        
        # Look for common patterns that indicate practical advice
        tip_indicators = [
            "protokol", "postup", "kroky", "návod", "odporúčanie", 
            "tipy", "rady", "ako", "môžete", "pomôže"
        ]
        
        sentences = content.split('. ')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(indicator in sentence for indicator in tip_indicators):
                if len(sentence) > 20 and len(sentence) < 150:
                    # Capitalize first letter and clean up
                    clean_sentence = sentence[0].upper() + sentence[1:]
                    if not clean_sentence.endswith('.'):
                        clean_sentence += '.'
                    tips.append(clean_sentence)
        
        return tips[:2]  # Max 2 tips
    
    def _get_no_results_response(self) -> str:
        """Response when no results are found."""
        return """
❌ **No relevant information found**

I couldn't find specific information about your question in the Slovak health and biology knowledge base.

💡 **Try asking about**:
• Mitochondria and cellular energy
• Epigenetics and gene expression
• Hormones and endocrine system
• Nutrition and metabolism
• Quantum biology concepts
• Circadian rhythms and light therapy
• Cold adaptation and thermogenesis

🔄 Try rephrasing your question or using different keywords.
        """.strip()
    
    def _get_low_confidence_response(self, query: str) -> str:
        """Response when confidence is low."""
        return f"""
⚠️ **Limited information found**

I found some potentially related content, but it may not directly answer your question about "{query}".

💡 **Suggestions**:
• Try using more specific Slovak health or biology terms
• Ask about specific topics like "mitochondria", "epigenetika", "hormóny"
• Rephrase your question with different keywords

🔄 The knowledge base focuses on Slovak content about health, biology, and quantum biology topics.
        """.strip()
    
    def display_search_results(self, results: List[Dict[str, Any]], show_details: bool = False):
        """Display search results in a formatted way."""
        print(f"\\n📋 Found {len(results)} relevant results:")
        print("-" * 60)
        
        for result in results:
            print(f"🔹 Rank #{result['rank']} (Similarity: {result['similarity']:.1%})")
            print(f"📄 Title: {result['metadata']['source_title']}")
            print(f"📅 Date: {result['metadata'].get('source_date', 'Unknown')}")
            
            if show_details:
                print(f"📝 Preview: {result['content'][:200]}...")
                print(f"🔗 URL: {result['metadata'].get('source_url', 'Unknown')}")
            
            print("-" * 40)
    
    def chat_loop(self):
        """Main chat interaction loop."""
        while True:
            try:
                # Get user input
                print("\\n" + "="*50)
                user_query = input("🧠 Your health question: ").strip()
                
                # Check for exit commands
                if user_query.lower() in ['quit', 'exit', 'bye', 'koniec', 'exit()']:
                    print("\\n👋 Goodbye! Stay healthy!")
                    break
                
                # Skip empty queries
                if not user_query:
                    print("❓ Please enter a question.")
                    continue
                
                # Handle special commands
                if user_query.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_query.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Process the query
                print("\\n" + "-"*50)
                start_time = time.time()
                
                # Search knowledge base
                search_results = self.search_knowledge_base(user_query, top_k=5)
                
                # Generate response
                response = self.generate_response(user_query, search_results)
                
                total_time = time.time() - start_time
                
                # Display response
                print("\\n🤖 **Slovak Health Chatbot Response:**")
                print("="*50)
                print(response)
                print("="*50)
                print(f"⏱️ Response generated in {total_time:.2f}s")
                
                # Optionally show search details
                show_details = input("\\n🔍 Show detailed search results? (y/N): ").lower().startswith('y')
                if show_details:
                    self.display_search_results(search_results, show_details=True)
            
            except KeyboardInterrupt:
                print("\\n\\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error processing your question: {e}")
                print("Please try again with a different question.")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🆘 **Slovak Health Chatbot Help**

**Available Commands:**
• Type any health/biology question in Slovak or English
• 'help' - Show this help message
• 'stats' - Show knowledge base statistics
• 'quit', 'exit', 'bye' - Exit the chatbot

**Example Questions:**
• "Čo sú mitochondrie?"
• "How does cold exposure affect metabolism?"
• "Epigenetika a výživa"
• "Circadian rhythms and light therapy"
• "Hormóny a spánok"

**Knowledge Base Topics:**
• Quantum Biology (Kvantová biológia)
• Mitochondria and cellular energy
• Epigenetics and gene expression
• Hormones and endocrine system
• Nutrition and metabolism
• Circadian rhythms and light therapy
• Cold adaptation and thermogenesis

**Tips for Better Results:**
• Use specific terms related to health and biology
• Try both Slovak and English terms
• Ask about specific protocols or mechanisms
• Reference specific topics like "leptín", "deutérium", "redox"
        """
        print(help_text)
    
    def _show_stats(self):
        """Show knowledge base statistics."""
        chunk_count = self.collection.count()
        print(f"""
📊 **Knowledge Base Statistics**

🗄️ Database: {self.db_path}
📁 Collection: {self.collection_name}
📄 Total chunks: {chunk_count:,}
🤖 Embedding model: {self.model_name}
📏 Embedding dimension: {self.model.get_sentence_embedding_dimension()}

📖 **Content Coverage:**
• 184 health and biology articles
• 604,509 total words
• Topics: mitochondria, epigenetics, hormones, nutrition, quantum biology
• Source: jaroslavlachky.sk (Slovak health expert)
        """)

def main():
    """Main function to run the chatbot."""
    parser = argparse.ArgumentParser(description="Slovak Health & Biology Knowledge Chatbot")
    parser.add_argument("--db-path", default="./vector_db", help="Path to vector database")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large", help="Embedding model name")
    parser.add_argument("--collection", default="slovak_blog_chunks", help="ChromaDB collection name")
    parser.add_argument("--query", help="Single query mode (non-interactive)")
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = SlovakHealthChatbot(
            db_path=args.db_path,
            model_name=args.model,
            collection_name=args.collection
        )
        
        # Single query mode
        if args.query:
            # Handle special commands in single query mode
            if args.query.lower() == 'help':
                chatbot._show_help()
                return
            if args.query.lower() == 'stats':
                chatbot._show_stats()
                return
            
            print(f"\\n🔍 Processing query: {args.query}")
            search_results = chatbot.search_knowledge_base(args.query, top_k=3)
            response = chatbot.generate_response(args.query, search_results)
            print("\\n🤖 Response:")
            print("="*50)
            print(response)
            return
        
        # Interactive chat mode
        chatbot.chat_loop()
        
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()