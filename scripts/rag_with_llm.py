#!/usr/bin/env python3
"""
RAG System with Qwen LLM Integration
Combines semantic search with Qwen2.5-7B for answer generation

Usage:
    # Interactive mode
    python rag_with_llm.py --interactive
    
    # Single query
    python rag_with_llm.py --query "How do I apply for housing?"
    
    # Custom model path
    python rag_with_llm.py --model-path ./models/qwen2.5-7b --query "What are the admission requirements?"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path to import rag_query
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGWithQwen:
    """
    Complete RAG system with Qwen LLM for answer generation.
    
    Architecture:
    1. Query → Semantic Search (FAISS)
    2. Retrieve top-k relevant chunks
    3. Build context from chunks
    4. Generate answer with Qwen LLM
    5. Return answer + sources
    """
    
    def __init__(self, model_path: str = "./models/qwen2.5-1.5b", embeddings_dir: str = "preprocessed/embeddings"):
        """
        Initialize RAG system with Qwen LLM.
        
        Args:
            model_path: Path to local Qwen model directory (default: 1.5B for faster local testing)
            embeddings_dir: Path to preprocessed embeddings
        """
        self.model_path = Path(model_path)
        self.embeddings_dir = embeddings_dir
        
        # Components
        self.rag_system = None
        self.tokenizer = None
        self.model = None
        
        # System prompt for THWS MAI assistant
        self.system_prompt = """You are a helpful assistant for THWS (Technische Hochschule Würzburg-Schweinfurt) Master of Artificial Intelligence (MAI) students.

Your role:
- Answer questions based ONLY on the provided context
- Be concise and clear
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Provide specific details when available (dates, requirements, contact information)
- Be friendly and supportive

Always prioritize accuracy over speculation."""
    
    def load_rag_system(self):
        """Load the RAG retrieval system."""
        logger.info("Loading RAG system...")
        
        try:
            from rag_query import CompleteProductionRAG
            self.rag_system = CompleteProductionRAG(embeddings_dir=self.embeddings_dir)
            self.rag_system.load_system()
            logger.info("✅ RAG system loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load RAG system: {e}")
            raise
    
    def load_qwen_model(self):
        """Load Qwen model from local directory."""
        logger.info(f"Loading Qwen model from: {self.model_path}")
        
        # Check if model exists locally
        if not self.model_path.exists():
            logger.error(f"❌ Model not found at {self.model_path}")
            logger.info("Please run: bash scripts/download_qwen.sh")
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            logger.info("Loading model (this may take a minute)...")
            logger.info("Using CPU for inference (faster than MPS for this model)")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",  # Force CPU usage
                trust_remote_code=True
            )
            
            # Get device info
            device = next(self.model.parameters()).device
            logger.info(f"✅ Qwen model loaded successfully on device: {device}")
            
        except ImportError:
            logger.error("❌ transformers library not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "accelerate"])
            logger.info("Please run the script again.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen model: {e}")
            raise
    
    def load_system(self):
        """Load both RAG and Qwen components."""
        logger.info("\n" + "="*60)
        logger.info("Initializing RAG with Qwen LLM System")
        logger.info("="*60 + "\n")
        
        self.load_rag_system()
        self.load_qwen_model()
        
        logger.info("\n✅ System ready!\n")
    
    def build_context(self, chunks: List[Dict], max_tokens: int = 1000) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks with text and metadata
            max_tokens: Maximum tokens for context (leave room for question + answer)
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source_file', 'Unknown')
            topic = chunk.get('topic', 'general')
            text = chunk.get('text', '')
            
            # Limit text length to avoid excessive context - increased for better answers
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            chunk_text = f"""[Source {i}: {source} | Topic: {topic}]
{text}
"""
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            
            # Stop if context is getting too large - increased for more comprehensive answers
            if current_length > 8000:  # ~2000 tokens
                break
        
        return "\n".join(context_parts)
    
    def build_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Build chat prompt for Qwen model.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
        
        Returns:
            Messages list in chat format
        """
        user_message = f"""Context from THWS MAI documentation:

{context}

Question: {question}

Please provide a helpful answer based on the context above."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def generate_answer(self, messages: List[Dict[str, str]], max_new_tokens: int = 300) -> str:
        """
        Generate answer using Qwen model.
        
        Args:
            messages: Chat messages in format [{"role": "system/user", "content": "..."}]
            max_new_tokens: Maximum tokens to generate (increased for complete answers)
        
        Returns:
            Generated answer text
        """
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate with CPU-optimized settings
            logger.info(f"Generating (max {max_new_tokens} tokens)...")
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for speed
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1,  # No beam search for speed
                repetition_penalty=1.1  # Prevent repetition
            )
            
            # Decode only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def query(self, question: str, top_k: int = 5, verbose: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            verbose: Print detailed logs
        
        Returns:
            Dictionary with question, answer, sources, and metadata
        """
        if not self.rag_system or not self.model:
            raise RuntimeError("System not loaded. Call load_system() first.")
        
        if verbose:
            logger.info("\n" + "="*60)
            logger.info(f"🔍 Question: {question}")
            logger.info("="*60)
        
        # Step 1: Retrieve relevant chunks
        if verbose:
            logger.info("\n📚 Retrieving relevant information...")
        
        rag_results = self.rag_system.query(question, top_k=top_k)
        chunks = rag_results['results']
        
        if verbose:
            logger.info(f"✅ Retrieved {len(chunks)} relevant chunks")
            logger.info(f"📊 Topics: {rag_results.get('topic_distribution', {})}")
        
        # Step 2: Build context
        context = self.build_context(chunks)
        
        if verbose:
            logger.info(f"\n📝 Context length: {len(context)} characters")
        
        # Step 3: Build prompt
        messages = self.build_prompt(question, context)
        
        # Step 4: Generate answer
        if verbose:
            logger.info("\n🤖 Generating answer with Qwen...")
        
        answer = self.generate_answer(messages)
        
        if verbose:
            logger.info("✅ Answer generated!\n")
            logger.info("="*60)
            logger.info("💬 ANSWER:")
            logger.info("="*60)
            print(f"\n{answer}\n")
            logger.info("="*60)
            logger.info("📖 SOURCES:")
            logger.info("="*60)
            for i, chunk in enumerate(chunks, 1):
                print(f"\n[{i}] {chunk.get('source_file', 'Unknown')} | Topic: {chunk.get('topic', 'general')}")
                print(f"    Similarity: {chunk.get('similarity_score', 0):.3f}")
            print("\n" + "="*60 + "\n")
        
        return {
            'question': question,
            'answer': answer,
            'sources': chunks,
            'rag_metadata': {
                'expanded_query': rag_results.get('expanded_query'),
                'disambiguated_query': rag_results.get('disambiguated_query'),
                'topic_distribution': rag_results.get('topic_distribution'),
                'num_chunks': len(chunks)
            }
        }
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        logger.info("\n" + "="*60)
        logger.info("🎓 THWS MAI RAG Chatbot - Interactive Mode")
        logger.info("="*60)
        logger.info("Ask questions about THWS MAI program!")
        logger.info("Commands: 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    logger.info("\n👋 Goodbye!")
                    break
                
                # Process query
                self.query(question, verbose=True)
                
            except KeyboardInterrupt:
                logger.info("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(
        description="RAG System with Qwen LLM for THWS MAI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/qwen2.5-1.5b',
        help='Path to Qwen model directory (default: ./models/qwen2.5-1.5b for fast local testing, use ./models/qwen2.5-7b for better quality)'
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='preprocessed/embeddings',
        help='Path to embeddings directory (default: preprocessed/embeddings)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of chunks to retrieve (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Create system
    rag_llm = RAGWithQwen(
        model_path=args.model_path,
        embeddings_dir=args.embeddings_dir
    )
    
    # Load system
    rag_llm.load_system()
    
    # Run based on mode
    if args.interactive:
        rag_llm.interactive_mode()
    elif args.query:
        rag_llm.query(args.query, top_k=args.top_k, verbose=True)
    else:
        logger.error("Please provide --query or --interactive")
        parser.print_help()


if __name__ == '__main__':
    main()
