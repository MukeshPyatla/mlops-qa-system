"""
Embedding pipeline for the LLM-Powered Q&A System.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .embedding_model import EmbeddingModel
from .vector_database import VectorDatabase
from ..utils.logging import get_logger, log_performance
from ..utils.text_processing import chunk_text
from ..utils.config import get_config


class EmbeddingPipeline:
    """
    Pipeline for creating and managing embeddings.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the embedding pipeline.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config
        self.logger = get_logger("embedding_pipeline")
        
        # Initialize components
        self.embedding_model = EmbeddingModel(config=config)
        self.vector_db = VectorDatabase(config=config)
        
        # Processing configuration
        self.processing_config = config.get("processing", {})
        self.chunk_size = self.processing_config.get("chunk_size", 1000)
        self.chunk_overlap = self.processing_config.get("chunk_overlap", 200)
        self.min_chunk_size = self.processing_config.get("min_chunk_size", 100)
        self.max_chunk_size = self.processing_config.get("max_chunk_size", 2000)
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         save_index: bool = True, 
                         index_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process documents through the embedding pipeline.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata' keys
            save_index: Whether to save the index to disk
            index_path: Path to save the index (auto-generated if None)
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting document processing", document_count=len(documents))
            
            # Extract and chunk documents
            chunks, chunk_metadata = self._chunk_documents(documents)
            
            # Create embeddings
            embeddings = self.embedding_model.encode(chunks)
            
            # Add to vector database
            self.vector_db.add_documents(chunks, embeddings, chunk_metadata)
            
            # Save index if requested
            saved_path = None
            if save_index:
                saved_path = self._save_index(index_path)
            
            duration = time.time() - start_time
            
            results = {
                "status": "success",
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "embedding_dimension": embeddings.shape[-1],
                "duration_seconds": duration,
                "index_path": saved_path,
                "index_stats": self.vector_db.get_index_stats()
            }
            
            self.logger.info(
                "Document processing completed",
                document_count=len(documents),
                chunk_count=len(chunks),
                duration_seconds=duration
            )
            
            log_performance(
                "document_processing",
                duration,
                document_count=len(documents),
                chunk_count=len(chunks)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Document processing failed",
                document_count=len(documents),
                error=str(e),
                exc_info=True
            )
            raise
    
    def _chunk_documents(self, documents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk documents into smaller pieces.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (chunks, chunk_metadata)
        """
        chunks = []
        chunk_metadata = []
        
        for doc_idx, document in enumerate(documents):
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            if not content:
                continue
            
            # Chunk the content
            document_chunks = chunk_text(
                content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=self.max_chunk_size
            )
            
            # Create metadata for each chunk
            for chunk_idx, chunk in enumerate(document_chunks):
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "document_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(document_chunks),
                    "chunk_size": len(chunk)
                })
                
                chunks.append(chunk)
                chunk_metadata.append(chunk_meta)
        
        self.logger.info(
            "Documents chunked",
            original_documents=len(documents),
            total_chunks=len(chunks)
        )
        
        return chunks, chunk_metadata
    
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)
            
            # Search vector database
            results = self.vector_db.search_with_metadata(query_embedding, k)
            
            self.logger.info(
                "Search completed",
                query=query,
                result_count=len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(
                "Search failed",
                query=query,
                error=str(e),
                exc_info=True
            )
            raise
    
    def load_index(self, index_path: str):
        """
        Load an existing index.
        
        Args:
            index_path: Path to the index file
        """
        try:
            self.vector_db.load(index_path)
            self.logger.info("Index loaded successfully", index_path=index_path)
            
        except Exception as e:
            self.logger.error(
                "Failed to load index",
                index_path=index_path,
                error=str(e),
                exc_info=True
            )
            raise
    
    def _save_index(self, index_path: Optional[str] = None) -> str:
        """
        Save the current index.
        
        Args:
            index_path: Path to save the index (auto-generated if None)
            
        Returns:
            Path where the index was saved
        """
        if index_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            index_path = f"models/faiss_index_{timestamp}"
        
        self.vector_db.save(index_path)
        return index_path
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding pipeline.
        
        Returns:
            Dictionary containing pipeline information
        """
        return {
            "embedding_model": self.embedding_model.get_model_info(),
            "vector_database": self.vector_db.get_index_stats(),
            "processing_config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size
            }
        }
    
    def clear_index(self):
        """Clear the current index."""
        self.vector_db.clear()
        self.logger.info("Index cleared")
    
    def batch_search(self, queries: List[str], k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        start_time = time.time()
        
        try:
            # Encode all queries
            query_embeddings = self.embedding_model.encode(queries)
            
            # Search for each query
            all_results = []
            for i, query in enumerate(queries):
                query_embedding = query_embeddings[i]
                results = self.vector_db.search_with_metadata(query_embedding, k)
                all_results.append(results)
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Batch search completed",
                query_count=len(queries),
                duration_seconds=duration
            )
            
            log_performance("batch_search", duration, query_count=len(queries))
            
            return all_results
            
        except Exception as e:
            self.logger.error(
                "Batch search failed",
                query_count=len(queries),
                error=str(e),
                exc_info=True
            )
            raise 