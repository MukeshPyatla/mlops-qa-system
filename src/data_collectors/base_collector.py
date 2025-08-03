"""
Base data collector class for the LLM-Powered Q&A System.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from ..utils.logging import get_logger, log_function_call
from ..utils.text_processing import clean_text, extract_metadata


class BaseDataCollector(ABC):
    """
    Abstract base class for data collectors.
    
    All data collectors should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the data collector.
        
        Args:
            name: Name of the collector
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = get_logger(f"collector.{name}")
        self.data_dir = Path(config.get("storage", {}).get("raw_data_dir", "data/raw"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    async def collect(self) -> List[Dict[str, Any]]:
        """
        Collect data from the source.
        
        Returns:
            List of collected data items
        """
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the data source.
        
        Returns:
            Dictionary containing source information
        """
        pass
    
    def save_data(self, data: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save collected data to disk.
        
        Args:
            data: List of data items to save
            filename: Optional filename, will generate one if not provided
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Add metadata to the data
        metadata = {
            "collector": self.name,
            "collected_at": datetime.now().isoformat(),
            "source_info": self.get_source_info(),
            "item_count": len(data)
        }
        
        output_data = {
            "metadata": metadata,
            "data": data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Data saved", filepath=str(filepath), item_count=len(data))
        return str(filepath)
    
    def load_data(self, filepath: str) -> Dict[str, Any]:
        """
        Load data from a saved file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Dictionary containing the loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info("Data loaded", filepath=filepath, item_count=len(data.get("data", [])))
        return data
    
    def process_text(self, text: str, **kwargs) -> str:
        """
        Process and clean text content.
        
        Args:
            text: Raw text to process
            **kwargs: Additional processing options
            
        Returns:
            Processed text
        """
        processing_config = self.config.get("processing", {})
        
        return clean_text(
            text,
            remove_html=processing_config.get("remove_html", True),
            remove_urls=processing_config.get("remove_urls", True),
            remove_emails=processing_config.get("remove_emails", True),
            normalize_whitespace=processing_config.get("normalize_whitespace", True)
        )
    
    def create_data_item(self, content: str, url: str = "", title: str = "", 
                         source: str = "", category: str = "", **kwargs) -> Dict[str, Any]:
        """
        Create a standardized data item.
        
        Args:
            content: Text content
            url: Source URL
            title: Document title
            source: Source name
            category: Content category
            **kwargs: Additional metadata
            
        Returns:
            Dictionary containing the data item
        """
        # Process the content
        processed_content = self.process_text(content)
        
        # Extract metadata
        metadata = extract_metadata(
            processed_content, url, title, source, category
        )
        
        # Add additional metadata
        metadata.update(kwargs)
        
        return {
            "content": processed_content,
            "metadata": metadata
        }
    
    async def collect_and_save(self, filename: Optional[str] = None) -> str:
        """
        Collect data and save it to disk.
        
        Args:
            filename: Optional filename for the saved data
            
        Returns:
            Path to the saved file
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting data collection", collector=self.name)
            
            # Collect data
            data = await self.collect()
            
            # Save data
            filepath = self.save_data(data, filename)
            
            duration = time.time() - start_time
            self.logger.info(
                "Data collection completed",
                collector=self.name,
                item_count=len(data),
                duration_seconds=duration,
                filepath=filepath
            )
            
            return filepath
            
        except Exception as e:
            self.logger.error(
                "Data collection failed",
                collector=self.name,
                error=str(e),
                exc_info=True
            )
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collector.
        
        Returns:
            Dictionary containing collector statistics
        """
        return {
            "name": self.name,
            "config": self.config,
            "data_dir": str(self.data_dir),
            "source_info": self.get_source_info()
        } 