"""
Vector database for the LLM-Powered Q&A System.
"""

import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

from ..utils.logging import get_logger, log_performance
from ..utils.config import get_config


class VectorDatabase:
    """
    FAISS-based vector database for similarity search.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the vector database.
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config.get("vector_db", {})
        self.index_type = self.config.get("index_type", "IndexFlatIP")
        self.dimension = self.config.get("dimension", 384)
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.normalize_L2 = self.config.get("normalize_L2", True)
        self.use_gpu = self.config.get("use_gpu", False)
        
        self.logger = get_logger("vector_database")
        
        # Initialize index
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index."""
        try:
            self.logger.info("Initializing FAISS index", index_type=self.index_type, dimension=self.dimension)
            
            if self.index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Move to GPU if requested and available
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("Index moved to GPU")
            
            self.logger.info("FAISS index initialized successfully")
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize FAISS index",
                error=str(e),
                exc_info=True
            )
            raise
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, 
                     metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents and their embeddings to the index.
        
        Args:
            documents: List of document texts
            embeddings: Array of document embeddings
            metadata: Optional list of metadata for each document
        """
        start_time = time.time()
        
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            self.logger.info(
                "Adding documents to index",
                document_count=len(documents),
                embedding_shape=embeddings.shape
            )
            
            # Normalize embeddings if required
            if self.normalize_L2:
                faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store documents and metadata
            self.documents.extend(documents)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                # Create default metadata
                default_metadata = [{"index": i} for i in range(len(documents))]
                self.metadata.extend(default_metadata)
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Documents added to index successfully",
                document_count=len(documents),
                total_documents=len(self.documents),
                duration_seconds=duration
            )
            
            log_performance(
                "document_indexing",
                duration,
                document_count=len(documents),
                total_documents=len(self.documents)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to add documents to index",
                document_count=len(documents),
                error=str(e),
                exc_info=True
            )
            raise
    
    def search(self, query_embedding: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return (defaults to self.top_k)
            
        Returns:
            Tuple of (distances, indices)
        """
        start_time = time.time()
        
        try:
            k = k or self.top_k
            
            # Normalize query embedding if required
            if self.normalize_L2:
                query_embedding = query_embedding.astype(np.float32)
                faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            # Search
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Search completed",
                k=k,
                duration_seconds=duration
            )
            
            log_performance("vector_search", duration, k=k)
            
            return distances[0], indices[0]
            
        except Exception as e:
            self.logger.error(
                "Search failed",
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_document(self, index: int) -> Tuple[str, Dict[str, Any]]:
        """
        Get document and metadata by index.
        
        Args:
            index: Document index
            
        Returns:
            Tuple of (document_text, metadata)
        """
        if index >= len(self.documents):
            raise IndexError(f"Document index {index} out of range")
        
        return self.documents[index], self.metadata[index]
    
    def get_documents(self, indices: List[int]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get multiple documents by indices.
        
        Args:
            indices: List of document indices
            
        Returns:
            List of (document_text, metadata) tuples
        """
        return [self.get_document(i) for i in indices]
    
    def search_with_metadata(self, query_embedding: np.ndarray, 
                           k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search and return results with metadata.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of result dictionaries with text, metadata, and similarity score
        """
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, (distance, index) in enumerate(zip(distances, indices)):
            if index < len(self.documents):  # Valid index
                document, metadata = self.get_document(index)
                
                # Convert distance to similarity score
                similarity = float(1.0 - distance) if self.index_type == "IndexFlatL2" else float(distance)
                
                result = {
                    "index": int(index),
                    "text": document,
                    "metadata": metadata,
                    "similarity": similarity,
                    "distance": float(distance),
                    "rank": i + 1
                }
                
                # Filter by similarity threshold
                if similarity >= self.similarity_threshold:
                    results.append(result)
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
        """
        return {
            "total_documents": len(self.documents),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "ntotal": self.index.ntotal
        }
    
    def save(self, filepath: str):
        """
        Save the index and documents to disk.
        
        Args:
            filepath: Path to save the index
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(filepath))
            
            # Save documents and metadata
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "config": self.config
            }
            
            with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("Index saved successfully", filepath=str(filepath))
            
        except Exception as e:
            self.logger.error(
                "Failed to save index",
                filepath=filepath,
                error=str(e),
                exc_info=True
            )
            raise
    
    def load(self, filepath: str):
        """
        Load the index and documents from disk.
        
        Args:
            filepath: Path to load the index from
        """
        try:
            filepath = Path(filepath)
            
            # Load FAISS index
            self.index = faiss.read_index(str(filepath))
            
            # Load documents and metadata
            with open(filepath.with_suffix('.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            
            self.logger.info(
                "Index loaded successfully",
                filepath=str(filepath),
                document_count=len(self.documents)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load index",
                filepath=filepath,
                error=str(e),
                exc_info=True
            )
            raise
    
    def clear(self):
        """Clear all documents from the index."""
        self._initialize_index()
        self.documents = []
        self.metadata = []
        self.logger.info("Index cleared") 