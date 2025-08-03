"""
API endpoints for the LLM-Powered Q&A System.
"""

import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import (
    QuestionRequest, QuestionResponse, BatchQuestionRequest, BatchQuestionResponse,
    SystemHealthResponse, SystemInfoResponse, DataCollectionRequest, DataCollectionResponse,
    IndexRebuildRequest, IndexRebuildResponse
)
from ..rag.rag_pipeline import RAGPipeline
from ..data_collectors.collector_manager import DataCollectorManager
from ..embedding.embedding_pipeline import EmbeddingPipeline
from ..utils.logging import get_logger, log_api_request
from ..utils.config import get_config

# Global variables for components
rag_pipeline = None
collector_manager = None
embedding_pipeline = None
start_time = time.time()

router = APIRouter()
logger = get_logger("api")


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


def get_collector_manager() -> DataCollectorManager:
    """Get or create collector manager instance."""
    global collector_manager
    if collector_manager is None:
        collector_manager = DataCollectorManager()
    return collector_manager


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get or create embedding pipeline instance."""
    global embedding_pipeline
    if embedding_pipeline is None:
        embedding_pipeline = EmbeddingPipeline()
    return embedding_pipeline


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a single question and get an answer.
    """
    start_time = time.time()
    
    try:
        logger.info("Received question request", question=request.question)
        
        # Get RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Answer the question
        result = pipeline.answer_question(request.question, request.k)
        
        # Convert to response model
        response = QuestionResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result["confidence"],
            retrieved_documents=result["retrieved_documents"],
            processing_time=result["processing_time"],
            question=result["question"],
            context_length=result["context_length"],
            error=result.get("error")
        )
        
        duration = time.time() - start_time
        log_api_request("POST", "/ask", 200, duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Error processing question", error=str(e), exc_info=True)
        log_api_request("POST", "/ask", 500, duration)
        
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/ask/batch", response_model=BatchQuestionResponse)
async def ask_batch_questions(request: BatchQuestionRequest):
    """
    Ask multiple questions and get answers.
    """
    start_time = time.time()
    
    try:
        logger.info("Received batch question request", question_count=len(request.questions))
        
        # Get RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Answer all questions
        results = pipeline.batch_answer_questions(request.questions, request.k)
        
        # Convert to response models
        answers = []
        for result in results:
            answer = QuestionResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                confidence=result["confidence"],
                retrieved_documents=result["retrieved_documents"],
                processing_time=result["processing_time"],
                question=result["question"],
                context_length=result["context_length"],
                error=result.get("error")
            )
            answers.append(answer)
        
        # Calculate summary statistics
        successful_answers = len([a for a in answers if a.error is None])
        total_processing_time = sum(a.processing_time for a in answers)
        
        response = BatchQuestionResponse(
            answers=answers,
            total_questions=len(request.questions),
            successful_answers=successful_answers,
            total_processing_time=total_processing_time
        )
        
        duration = time.time() - start_time
        log_api_request("POST", "/ask/batch", 200, duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Error processing batch questions", error=str(e), exc_info=True)
        log_api_request("POST", "/ask/batch", 500, duration)
        
        raise HTTPException(status_code=500, detail=f"Error processing batch questions: {str(e)}")


@router.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Check system health and status.
    """
    try:
        # Get component instances
        pipeline = get_rag_pipeline()
        collector_mgr = get_collector_manager()
        embedding_pipe = get_embedding_pipeline()
        
        # Check component health
        components = {
            "rag_pipeline": "healthy",
            "collector_manager": "healthy",
            "embedding_pipeline": "healthy"
        }
        
        # Get system info
        pipeline_info = pipeline.get_pipeline_info()
        collector_stats = collector_mgr.get_collector_stats()
        embedding_info = embedding_pipe.get_pipeline_info()
        
        # Calculate uptime
        uptime = time.time() - start_time
        
        response = SystemHealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=uptime,
            components={
                **components,
                "pipeline_info": pipeline_info,
                "collector_stats": collector_stats,
                "embedding_info": embedding_info
            }
        )
        
        log_api_request("GET", "/health", 200, 0.0)
        return response
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        log_api_request("GET", "/health", 500, 0.0)
        
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/info", response_model=SystemInfoResponse)
async def system_info():
    """
    Get detailed system information.
    """
    try:
        # Get component instances
        pipeline = get_rag_pipeline()
        collector_mgr = get_collector_manager()
        embedding_pipe = get_embedding_pipeline()
        
        # Get system information
        system_info = {
            "version": "1.0.0",
            "uptime": time.time() - start_time,
            "available_sources": collector_mgr.get_available_sources(),
            "configuration": get_config("models")
        }
        
        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        
        # Get index statistics
        index_stats = embedding_pipe.vector_db.get_index_stats()
        
        # Get model information
        model_info = {
            "embedding_model": embedding_pipe.embedding_model.get_model_info(),
            "llm_model": pipeline.llm_model.get_model_info()
        }
        
        response = SystemInfoResponse(
            system_info=system_info,
            pipeline_info=pipeline_info,
            index_stats=index_stats,
            model_info=model_info
        )
        
        log_api_request("GET", "/info", 200, 0.0)
        return response
        
    except Exception as e:
        logger.error("Failed to get system info", error=str(e), exc_info=True)
        log_api_request("GET", "/info", 500, 0.0)
        
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@router.post("/collect-data", response_model=DataCollectionResponse)
async def collect_data(request: DataCollectionRequest):
    """
    Collect data from configured sources.
    """
    start_time = time.time()
    
    try:
        logger.info("Received data collection request")
        
        # Get collector manager
        collector_mgr = get_collector_manager()
        
        # Collect data
        if request.sources:
            # Collect from specific sources
            results = {}
            for source in request.sources:
                try:
                    result = await collector_mgr.collect_from_source(source, request.save_results)
                    results[source] = result
                except Exception as e:
                    logger.error(f"Failed to collect from {source}", error=str(e))
                    results[source] = {"status": "failed", "error": str(e)}
        else:
            # Collect from all sources
            results = await collector_mgr.collect_all_data(request.save_results)
        
        response = DataCollectionResponse(
            status="completed",
            results=results
        )
        
        duration = time.time() - start_time
        log_api_request("POST", "/collect-data", 200, duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Error collecting data", error=str(e), exc_info=True)
        log_api_request("POST", "/collect-data", 500, duration)
        
        raise HTTPException(status_code=500, detail=f"Error collecting data: {str(e)}")


@router.post("/rebuild-index", response_model=IndexRebuildResponse)
async def rebuild_index(request: IndexRebuildRequest):
    """
    Rebuild the vector index from collected data.
    """
    start_time = time.time()
    
    try:
        logger.info("Received index rebuild request")
        
        # Get embedding pipeline
        embedding_pipe = get_embedding_pipeline()
        
        # Clear existing index if requested
        if request.clear_existing:
            embedding_pipe.clear_index()
        
        # TODO: Implement data loading and processing
        # This would involve loading collected data files and processing them
        
        results = {
            "status": "completed",
            "message": "Index rebuild completed (placeholder implementation)"
        }
        
        response = IndexRebuildResponse(
            status="completed",
            results=results
        )
        
        duration = time.time() - start_time
        log_api_request("POST", "/rebuild-index", 200, duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Error rebuilding index", error=str(e), exc_info=True)
        log_api_request("POST", "/rebuild-index", 500, duration)
        
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


@router.get("/")
async def root():
    """
    Root endpoint with basic information.
    """
    return {
        "message": "LLM-Powered Multi-Source Q&A System",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "ask": "/ask",
            "batch_ask": "/ask/batch",
            "health": "/health",
            "info": "/info",
            "collect_data": "/collect-data",
            "rebuild_index": "/rebuild-index"
        }
    } 