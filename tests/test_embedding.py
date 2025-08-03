"""
Unit tests for embedding components.
"""

import pytest
import numpy as np
from src.embedding.embedding_model import EmbeddingModel
from src.embedding.vector_database import VectorDatabase


class TestEmbeddingModel:
    """Test embedding model functionality."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for testing."""
        # Use a smaller model for testing
        return EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def test_embedding_model_initialization(self, embedding_model):
        """Test embedding model initialization."""
        assert embedding_model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedding_model.dimension == 384
        assert embedding_model.batch_size == 32
    
    def test_encode_single_text(self, embedding_model):
        """Test encoding a single text."""
        text = "This is a test sentence."
        embedding = embedding_model.encode_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.isnan(embedding).any()
    
    def test_encode_multiple_texts(self, embedding_model):
        """Test encoding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedding_model.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.isnan(embeddings).any()
    
    def test_similarity_calculation(self, embedding_model):
        """Test similarity calculation."""
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        
        embedding1 = embedding_model.encode_single(text1)
        embedding2 = embedding_model.encode_single(text2)
        
        similarity = embedding_model.similarity(embedding1, embedding2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_batch_similarity(self, embedding_model):
        """Test batch similarity calculation."""
        query_text = "Query sentence."
        candidate_texts = ["First candidate.", "Second candidate.", "Third candidate."]
        
        query_embedding = embedding_model.encode_single(query_text)
        candidate_embeddings = embedding_model.encode(candidate_texts)
        
        similarities = embedding_model.batch_similarity(query_embedding, candidate_embeddings)
        
        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (3,)
        assert all(0 <= sim <= 1 for sim in similarities)


class TestVectorDatabase:
    """Test vector database functionality."""
    
    @pytest.fixture
    def vector_db(self):
        """Create vector database for testing."""
        return VectorDatabase()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            "This is the first document about machine learning.",
            "The second document discusses artificial intelligence.",
            "Document three covers natural language processing.",
            "This document is about computer vision and image recognition.",
            "The final document talks about deep learning and neural networks."
        ]
    
    @pytest.fixture
    def sample_embeddings(self, embedding_model, sample_documents):
        """Create sample embeddings for testing."""
        return embedding_model.encode(sample_documents)
    
    def test_vector_database_initialization(self, vector_db):
        """Test vector database initialization."""
        assert vector_db.index_type == "IndexFlatIP"
        assert vector_db.dimension == 384
        assert vector_db.top_k == 5
        assert vector_db.similarity_threshold == 0.7
    
    def test_add_documents(self, vector_db, sample_documents, sample_embeddings):
        """Test adding documents to the index."""
        vector_db.add_documents(sample_documents, sample_embeddings)
        
        assert len(vector_db.documents) == len(sample_documents)
        assert len(vector_db.metadata) == len(sample_documents)
        assert vector_db.index.ntotal == len(sample_documents)
    
    def test_search(self, vector_db, sample_documents, sample_embeddings):
        """Test searching the index."""
        vector_db.add_documents(sample_documents, sample_embeddings)
        
        query = "machine learning"
        query_embedding = vector_db.embedding_model.encode_single(query)
        
        distances, indices = vector_db.search(query_embedding, k=3)
        
        assert len(distances) == 3
        assert len(indices) == 3
        assert all(0 <= idx < len(sample_documents) for idx in indices)
    
    def test_search_with_metadata(self, vector_db, sample_documents, sample_embeddings):
        """Test searching with metadata."""
        metadata = [
            {"source": "test", "title": f"Doc {i}"} 
            for i in range(len(sample_documents))
        ]
        
        vector_db.add_documents(sample_documents, sample_embeddings, metadata)
        
        query = "artificial intelligence"
        query_embedding = vector_db.embedding_model.encode_single(query)
        
        results = vector_db.search_with_metadata(query_embedding, k=3)
        
        assert len(results) <= 3
        for result in results:
            assert "text" in result
            assert "metadata" in result
            assert "similarity" in result
            assert result["similarity"] >= vector_db.similarity_threshold
    
    def test_get_document(self, vector_db, sample_documents, sample_embeddings):
        """Test getting document by index."""
        vector_db.add_documents(sample_documents, sample_embeddings)
        
        document, metadata = vector_db.get_document(0)
        
        assert document == sample_documents[0]
        assert isinstance(metadata, dict)
    
    def test_get_index_stats(self, vector_db, sample_documents, sample_embeddings):
        """Test getting index statistics."""
        vector_db.add_documents(sample_documents, sample_embeddings)
        
        stats = vector_db.get_index_stats()
        
        assert stats["total_documents"] == len(sample_documents)
        assert stats["index_type"] == "IndexFlatIP"
        assert stats["dimension"] == 384
        assert stats["ntotal"] == len(sample_documents) 