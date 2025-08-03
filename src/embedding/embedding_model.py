"""
Embedding model for the LLM-Powered Q&A System.
"""

import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import time

from ..utils.logging import get_logger, log_performance
from ..utils.config import get_config


class EmbeddingModel:
    """
    BERT-based embedding model for text vectorization.
    """
    
    def __init__(self, model_name: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use
            config: Configuration dictionary
        """
        if config is None:
            config = get_config("models")
        
        self.config = config.get("embedding", {})
        self.model_name = model_name or self.config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.config.get("dimension", 384)
        self.max_length = self.config.get("max_length", 512)
        self.device = self.config.get("device", "auto")
        self.batch_size = self.config.get("batch_size", 32)
        
        self.logger = get_logger("embedding_model")
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BERT model."""
        try:
            self.logger.info("Loading embedding model", model=self.model_name)
            
            # Use sentence-transformers for easier handling
            self.model = SentenceTransformer(self.model_name)
            
            # Set device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model.to(self.device)
            
            # Test the model
            test_embedding = self.model.encode("test", convert_to_tensor=True)
            self.logger.info(
                "Embedding model loaded successfully",
                model=self.model_name,
                device=self.device,
                dimension=test_embedding.shape[-1]
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to load embedding model",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True, 
               show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Text or list of texts to encode
            normalize: Whether to normalize embeddings
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        start_time = time.time()
        
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
            
            self.logger.info(
                "Encoding texts",
                text_count=len(texts),
                batch_size=self.batch_size
            )
            
            # Encode texts
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            
            duration = time.time() - start_time
            
            self.logger.info(
                "Texts encoded successfully",
                text_count=len(texts),
                embedding_shape=embeddings.shape,
                duration_seconds=duration
            )
            
            log_performance(
                "text_encoding",
                duration,
                text_count=len(texts),
                embedding_dimension=embeddings.shape[-1]
            )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(
                "Failed to encode texts",
                text_count=len(texts) if isinstance(texts, list) else 1,
                error=str(e),
                exc_info=True
            )
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            normalize: Whether to normalize the embedding
            
        Returns:
            Embedding vector
        """
        embeddings = self.encode([text], normalize=normalize, show_progress_bar=False)
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.dimension
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between a query and multiple candidates.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize candidate embeddings
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarities
        similarities = np.dot(candidate_norms, query_norm)
        
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "device": self.device,
            "batch_size": self.batch_size
        } 