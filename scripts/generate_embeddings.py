#!/usr/bin/env python3
"""
Generate embeddings for preprocessed chunks and create FAISS index for RAG system.

Supports multilingual content (German/English) and provides fast similarity search
for the THWS MAI RAG system.

Features:
- Multilingual embedding models (handles German + English)
- FAISS indexing for fast retrieval
- Batch processing for memory efficiency
- Progress tracking and statistics
- Topic-aware indexing option

Usage:
    python scripts/generate_embeddings.py --input preprocessed/rag_chunks.jsonl --output preprocessed/embeddings
    
    # Use specific model
    python scripts/generate_embeddings.py --model BAAI/bge-small-en --input preprocessed/rag_chunks.jsonl
    
    # Topic-specific embeddings
    python scripts/generate_embeddings.py --by-topic --input preprocessed/chunks_by_topic

Outputs:
    - embeddings/embeddings.index (FAISS index)
    - embeddings/chunk_metadata.json (chunk info for retrieval)
    - embeddings/model_info.json (model and processing details)
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", batch_size: int = 32):
        """
        Initialize embedding generator with multilingual support.
        
        Args:
            model_name: Embedding model to use
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        
        # Model recommendations based on multilingual capability
        self.multilingual_models = {
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "description": "Multilingual, 384-dim, good for German+English",
                "dimension": 384,
                "languages": ["en", "de", "fr", "es", "it", "zh", "ja", "ko"]
            },
            "BAAI/bge-small-en": {
                "description": "English-focused, 384-dim, high quality",
                "dimension": 384,
                "languages": ["en"]
            },
            "BAAI/bge-small-en-v1.5": {
                "description": "English-focused, improved version",
                "dimension": 384,
                "languages": ["en"]
            },
            "intfloat/multilingual-e5-small": {
                "description": "Multilingual E5, 384-dim, excellent for mixed languages",
                "dimension": 384,
                "languages": ["en", "de", "fr", "es", "it", "zh", "ja", "ar", "hi"]
            },
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
                "description": "Multilingual, 768-dim, higher quality but larger",
                "dimension": 768,
                "languages": ["en", "de", "fr", "es", "it", "zh", "ja", "ko"]
            }
        }
        
    def load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Check if model is in our known list
            if self.model_name in self.multilingual_models:
                model_info = self.multilingual_models[self.model_name]
                logger.info(f"Model info: {model_info['description']}")
                logger.info(f"Supported languages: {', '.join(model_info['languages'])}")
                self.embedding_dim = model_info['dimension']
            
            self.model = SentenceTransformer(self.model_name)
            
            # Get actual embedding dimension if not known
            if self.embedding_dim is None:
                test_embedding = self.model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
                
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def detect_language_distribution(self, texts: List[str], sample_size: int = 1000) -> Dict[str, float]:
        """
        Detect language distribution in the text corpus.
        
        Args:
            texts: List of text chunks
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary with language distribution
        """
        try:
            import langdetect
            from collections import Counter
            
            # Sample texts for language detection
            sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
            
            detected_langs = []
            for text in sample_texts:
                try:
                    # Use first 200 chars for detection (more reliable)
                    sample_text = text[:200].strip()
                    if len(sample_text) > 10:
                        lang = langdetect.detect(sample_text)
                        detected_langs.append(lang)
                except:
                    detected_langs.append('unknown')
            
            # Calculate distribution
            lang_counts = Counter(detected_langs)
            total = len(detected_langs)
            
            distribution = {lang: count/total for lang, count in lang_counts.items()}
            return distribution
            
        except ImportError:
            logger.warning("langdetect not available. Install with: pip install langdetect")
            return {"unknown": 1.0}
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {"unknown": 1.0}
    
    def load_chunks(self, input_path: Path) -> Tuple[List[Dict], List[str]]:
        """
        Load chunks from JSONL file.
        
        Args:
            input_path: Path to JSONL file
            
        Returns:
            Tuple of (chunk_metadata, texts)
        """
        chunks = []
        texts = []
        
        logger.info(f"Loading chunks from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                    texts.append(chunk['text'])
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks, texts
    
    def load_chunks_by_topic(self, topic_dir: Path) -> Dict[str, Tuple[List[Dict], List[str]]]:
        """
        Load chunks organized by topic.
        
        Args:
            topic_dir: Directory containing topic subdirectories
            
        Returns:
            Dictionary mapping topic to (chunk_metadata, texts)
        """
        topic_data = {}
        
        for topic_path in topic_dir.iterdir():
            if topic_path.is_dir():
                topic_name = topic_path.name
                jsonl_file = topic_path / f"{topic_name}_chunks.jsonl"
                
                if jsonl_file.exists():
                    chunks, texts = self.load_chunks(jsonl_file)
                    topic_data[topic_name] = (chunks, texts)
                    logger.info(f"Loaded {len(chunks)} chunks for topic: {topic_name}")
        
        return topic_data
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            self.load_model()
        
        # Process in batches to manage memory
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> 'faiss.Index':
        """
        Create FAISS index for fast similarity search.
        
        Args:
            embeddings: Embedding vectors
            use_gpu: Whether to use GPU for indexing
            
        Returns:
            FAISS index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        logger.info(f"Creating FAISS index for {n_vectors} vectors of dimension {dimension}")
        
        # Choose index type based on number of vectors
        if n_vectors < 50000:
            # For smaller datasets, use flat index (exact search)
            index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity with normalized vectors)
            logger.info("Using IndexFlatIP (exact search)")
        else:
            # For larger datasets, use IVF index (approximate search)
            nlist = min(int(np.sqrt(n_vectors)), n_vectors // 100)  # Conservative number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(f"Using IndexIVFFlat with {nlist} clusters (approximate search)")
            
            # Ensure we have enough training points
            if n_vectors < nlist * 39:  # FAISS recommendation: 39 * nlist training points
                logger.warning(f"Not enough vectors for IVF training. Using flat index instead.")
                index = faiss.IndexFlatIP(dimension)
                logger.info("Switched to IndexFlatIP (exact search)")
            else:
                # Train the index
                logger.info("Training index...")
                index.train(embeddings.astype('float32'))
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings.astype('float32'))
        
        # Use GPU if available and requested
        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Moved index to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        
        return index
    
    def save_embeddings(self, 
                       embeddings: np.ndarray, 
                       chunks: List[Dict], 
                       output_dir: Path,
                       topic_name: str = None,
                       language_dist: Dict = None):
        """
        Save embeddings, FAISS index, and metadata.
        
        Args:
            embeddings: Generated embeddings
            chunks: Original chunk metadata
            output_dir: Output directory
            topic_name: Topic name (if topic-specific)
            language_dist: Language distribution info
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create FAISS index
        index = self.create_faiss_index(embeddings)
        
        # Save FAISS index
        import faiss
        index_file = output_dir / f"embeddings{'_' + topic_name if topic_name else ''}.index"
        faiss.write_index(index, str(index_file))
        logger.info(f"Saved FAISS index to {index_file}")
        
        # Save chunk metadata for retrieval
        metadata_file = output_dir / f"chunk_metadata{'_' + topic_name if topic_name else ''}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunk metadata to {metadata_file}")
        
        # Save embeddings as numpy array (optional, for backup)
        embeddings_file = output_dir / f"embeddings_array{'_' + topic_name if topic_name else ''}.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings array to {embeddings_file}")
        
        # Save processing info
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "num_chunks": len(chunks),
            "processing_date": datetime.now().isoformat(),
            "batch_size": self.batch_size,
            "topic": topic_name,
            "language_distribution": language_dist,
            "index_type": type(index).__name__,
            "embedding_shape": embeddings.shape
        }
        
        info_file = output_dir / f"model_info{'_' + topic_name if topic_name else ''}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved model info to {info_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for THWS MAI RAG system')
    parser.add_argument('--input', required=True, help='Input JSONL file or topic directory')
    parser.add_argument('--output', default='preprocessed/embeddings', help='Output directory')
    parser.add_argument('--model', default='intfloat/multilingual-e5-small', 
                       help='Embedding model (default: multilingual-e5-small for German+English)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--by-topic', action='store_true', help='Generate topic-specific embeddings')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for FAISS index')
    parser.add_argument('--detect-lang', action='store_true', help='Detect language distribution')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name=args.model, batch_size=args.batch_size)
    
    logger.info("=" * 60)
    logger.info("THWS MAI RAG - EMBEDDING GENERATION")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    
    if args.by_topic and input_path.is_dir():
        # Process by topic
        topic_data = generator.load_chunks_by_topic(input_path)
        
        for topic_name, (chunks, texts) in topic_data.items():
            logger.info(f"\n--- Processing topic: {topic_name} ---")
            
            # Detect language distribution if requested
            language_dist = None
            if args.detect_lang:
                language_dist = generator.detect_language_distribution(texts)
                logger.info(f"Language distribution: {language_dist}")
            
            # Generate embeddings
            embeddings = generator.generate_embeddings_batch(texts)
            
            # Save results
            generator.save_embeddings(
                embeddings, chunks, output_dir, 
                topic_name=topic_name, language_dist=language_dist
            )
            
            logger.info(f"Completed topic: {topic_name} ({len(chunks)} chunks)")
    
    else:
        # Process single file
        if input_path.is_dir():
            input_path = input_path / "rag_chunks.jsonl"
        
        chunks, texts = generator.load_chunks(input_path)
        
        # Detect language distribution if requested
        language_dist = None
        if args.detect_lang:
            language_dist = generator.detect_language_distribution(texts)
            logger.info(f"Language distribution: {language_dist}")
        
        # Generate embeddings
        embeddings = generator.generate_embeddings_batch(texts)
        
        # Save results
        generator.save_embeddings(embeddings, chunks, output_dir, language_dist=language_dist)
    
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("Files created:")
    logger.info("  - embeddings.index (FAISS index)")
    logger.info("  - chunk_metadata.json (chunk info)")
    logger.info("  - embeddings_array.npy (embeddings backup)")
    logger.info("  - model_info.json (processing details)")

if __name__ == '__main__':
    main()