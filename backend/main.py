#!/usr/bin/env python3
"""
FastAPI Backend for THWS MAI RAG Chatbot
Provides REST API endpoints for question answering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.rag_with_llm import RAGWithQwen

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="THWS MAI Chatbot API",
    description="Question answering system for THWS Master of Artificial Intelligence students",
    version="1.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RAGWithQwen] = None


class QuestionRequest(BaseModel):
    """Request model for asking questions"""
    question: str
    top_k: int = 5  # Retrieve top 5 chunks for better context


class Source(BaseModel):
    """Source document information"""
    source_file: str
    topic: str
    similarity_score: float
    text: str


class AnswerResponse(BaseModel):
    """Response model with answer and sources"""
    question: str
    answer: str
    sources: List[Source]
    metadata: Dict


@app.on_event("startup")
async def startup_event():
    """Load RAG system on startup"""
    global rag_system
    logger.info("="*60)
    logger.info("Starting THWS MAI Chatbot Backend")
    logger.info("="*60)
    
    try:
        # Initialize RAG system
        logger.info("Loading RAG system with Qwen LLM...")
        rag_system = RAGWithQwen(
            model_path="./models/qwen2.5-1.5b",  # Use faster 1.5B model
            embeddings_dir="preprocessed/embeddings"
        )
        rag_system.load_system()
        logger.info("✅ RAG system loaded successfully!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"❌ Failed to load RAG system: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "THWS MAI Chatbot API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_loaded": rag_system is not None,
        "model": "Qwen2.5-1.5B-Instruct"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG system
    
    Args:
        request: QuestionRequest with question text and top_k
        
    Returns:
        AnswerResponse with answer, sources, and metadata
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"Received question: {request.question}")
        
        # Get answer from RAG system
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            verbose=False  # Disable verbose logging for API
        )
        
        # Format sources
        sources = [
            Source(
                source_file=chunk.get('source_file', 'Unknown'),
                topic=chunk.get('topic', 'general'),
                similarity_score=chunk.get('similarity_score', 0.0),
                text=chunk.get('text', '')[:500] + "..."  # Truncate for API response
            )
            for chunk in result['sources']
        ]
        
        response = AnswerResponse(
            question=result['question'],
            answer=result['answer'],
            sources=sources,
            metadata=result.get('rag_metadata', {})
        )
        
        logger.info("✅ Answer generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        return {
            "total_chunks": len(rag_system.rag_system.chunks) if rag_system.rag_system else 0,
            "embedding_model": "intfloat/multilingual-e5-small",
            "llm_model": "Qwen2.5-1.5B-Instruct",
            "topics": ["housing", "admission", "course-info", "wurzburg-guide", "general"]
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
