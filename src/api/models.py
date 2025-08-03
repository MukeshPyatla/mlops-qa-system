"""
Pydantic models for the LLM-Powered Q&A System API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class QuestionRequest(BaseModel):
    """Request model for a single question."""
    
    question: str = Field(..., description="The question to answer", min_length=1, max_length=1000)
    k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    include_sources: bool = Field(True, description="Whether to include source information in response")
    temperature: Optional[float] = Field(0.7, description="Temperature for text generation", ge=0.0, le=2.0)


class SourceInfo(BaseModel):
    """Information about a source document."""
    
    source: str = Field(..., description="Source name")
    title: str = Field(..., description="Document title")
    url: str = Field("", description="Source URL")
    similarity: float = Field(..., description="Similarity score", ge=0.0, le=1.0)


class QuestionResponse(BaseModel):
    """Response model for a single question."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    retrieved_documents: int = Field(..., description="Number of documents retrieved")
    processing_time: float = Field(..., description="Processing time in seconds")
    question: str = Field(..., description="Original question")
    context_length: int = Field(..., description="Length of context used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if any")


class BatchQuestionRequest(BaseModel):
    """Request model for multiple questions."""
    
    questions: List[str] = Field(..., description="List of questions to answer", min_items=1, max_items=10)
    k: Optional[int] = Field(5, description="Number of documents to retrieve per question", ge=1, le=20)
    include_sources: bool = Field(True, description="Whether to include source information in response")
    temperature: Optional[float] = Field(0.7, description="Temperature for text generation", ge=0.0, le=2.0)


class BatchQuestionResponse(BaseModel):
    """Response model for multiple questions."""
    
    answers: List[QuestionResponse] = Field(..., description="List of answers")
    total_questions: int = Field(..., description="Total number of questions")
    successful_answers: int = Field(..., description="Number of successful answers")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class SystemHealthResponse(BaseModel):
    """Response model for system health check."""
    
    status: str = Field(..., description="System status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    components: Dict[str, Any] = Field(..., description="Component health information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    
    system_info: Dict[str, Any] = Field(..., description="System information")
    pipeline_info: Dict[str, Any] = Field(..., description="Pipeline information")
    index_stats: Dict[str, Any] = Field(..., description="Index statistics")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Info timestamp")


class DataCollectionRequest(BaseModel):
    """Request model for data collection."""
    
    sources: Optional[List[str]] = Field(None, description="Specific sources to collect from")
    save_results: bool = Field(True, description="Whether to save collection results")
    force_refresh: bool = Field(False, description="Whether to force refresh existing data")


class DataCollectionResponse(BaseModel):
    """Response model for data collection."""
    
    status: str = Field(..., description="Collection status")
    results: Dict[str, Any] = Field(..., description="Collection results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Collection timestamp")


class IndexRebuildRequest(BaseModel):
    """Request model for index rebuilding."""
    
    data_files: Optional[List[str]] = Field(None, description="Specific data files to process")
    save_index: bool = Field(True, description="Whether to save the new index")
    clear_existing: bool = Field(False, description="Whether to clear existing index")


class IndexRebuildResponse(BaseModel):
    """Response model for index rebuilding."""
    
    status: str = Field(..., description="Rebuild status")
    results: Dict[str, Any] = Field(..., description="Rebuild results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Rebuild timestamp") 