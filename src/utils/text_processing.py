"""
Text processing utilities for the LLM-Powered Q&A System.
"""

import re
import html
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib


def clean_text(text: str, remove_html: bool = True, remove_urls: bool = True, 
               remove_emails: bool = True, normalize_whitespace: bool = True) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        remove_html: Whether to remove HTML tags
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    if remove_emails:
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200,
               min_chunk_size: int = 100, max_chunk_size: int = 2000) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    # Split by sentences first to maintain coherence
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it meets minimum size
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    # If we have no chunks or only one chunk, split by words
    if len(chunks) <= 1:
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-chunk_overlap//10:]  # Approximate overlap
                current_chunk = " ".join(overlap_words) + " " + word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks


def extract_metadata(text: str, url: str = "", title: str = "", 
                    source: str = "", category: str = "") -> Dict[str, Any]:
    """
    Extract metadata from text and provided information.
    
    Args:
        text: The text content
        url: Source URL
        title: Document title
        source: Source name
        category: Content category
        
    Returns:
        Dictionary containing metadata
    """
    metadata = {
        "source": source,
        "url": url,
        "title": title,
        "category": category,
        "text_length": len(text),
        "word_count": len(text.split()),
        "extracted_at": datetime.now().isoformat(),
        "content_hash": hashlib.md5(text.encode()).hexdigest()
    }
    
    # Extract date if present in text
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'  # DD MMM YYYY
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            metadata["extracted_date"] = matches[0]
            break
    
    # Extract language (simple heuristic)
    if re.search(r'[а-яА-Я]', text):
        metadata["language"] = "russian"
    elif re.search(r'[а-яА-Я]', text):
        metadata["language"] = "chinese"
    else:
        metadata["language"] = "english"
    
    return metadata


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Text to analyze
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Clean and tokenize
    text = clean_text(text.lower())
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Clean and tokenize
    text1 = set(clean_text(text1.lower()).split())
    text2 = set(clean_text(text2.lower()).split())
    
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    intersection = text1.intersection(text2)
    union = text1.union(text2)
    
    return len(intersection) / len(union)


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
    
    return text.strip() 