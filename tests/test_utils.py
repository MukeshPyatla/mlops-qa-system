"""
Unit tests for utility functions.
"""

import pytest
from src.utils.text_processing import clean_text, chunk_text, extract_metadata
from src.utils.config import load_config, get_config


class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test HTML removal
        html_text = "<p>This is <b>bold</b> text</p>"
        cleaned = clean_text(html_text)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "This is bold text" in cleaned
        
        # Test URL removal
        url_text = "Check out https://example.com for more info"
        cleaned = clean_text(url_text)
        assert "https://example.com" not in cleaned
        
        # Test email removal
        email_text = "Contact us at user@example.com"
        cleaned = clean_text(email_text)
        assert "user@example.com" not in cleaned
        
        # Test whitespace normalization
        whitespace_text = "  multiple    spaces  "
        cleaned = clean_text(whitespace_text)
        assert cleaned == "multiple spaces"
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        long_text = "This is a long text. " * 50
        
        chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + overlap
        
        # Test minimum chunk size
        short_text = "Short text"
        chunks = chunk_text(short_text, min_chunk_size=50)
        assert len(chunks) == 0  # Too short
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        text = "This is a test document from 2024-01-15"
        metadata = extract_metadata(
            text,
            url="https://example.com",
            title="Test Document",
            source="test",
            category="test"
        )
        
        assert metadata["source"] == "test"
        assert metadata["title"] == "Test Document"
        assert metadata["url"] == "https://example.com"
        assert metadata["category"] == "test"
        assert metadata["text_length"] == len(text)
        assert "content_hash" in metadata


class TestConfig:
    """Test configuration utilities."""
    
    def test_load_config(self):
        """Test configuration loading."""
        # This test would require a test config file
        # For now, we'll test that the function exists
        assert callable(load_config)
        assert callable(get_config)
    
    def test_get_config(self):
        """Test getting configuration by name."""
        # This test would require test config files
        # For now, we'll test that the function exists
        assert callable(get_config) 