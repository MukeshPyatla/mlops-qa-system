"""
Data collector manager for the LLM-Powered Q&A System.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from .documentation_collector import DocumentationCollector
from .wikipedia_collector import WikipediaCollector
from .news_collector import NewsCollector
from .github_collector import GitHubCollector
from ..utils.logging import get_logger, log_performance
from ..utils.config import get_config


class DataCollectorManager:
    """
    Manager for coordinating all data collectors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data collector manager.
        
        Args:
            config: Configuration dictionary, will load from file if not provided
        """
        if config is None:
            config = get_config("data_sources")
        
        self.config = config
        self.logger = get_logger("collector_manager")
        self.collectors = {}
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _initialize_collectors(self):
        """Initialize all data collectors."""
        # Documentation collector
        if self.config.get("sources", {}).get("documentation"):
            self.collectors["documentation"] = DocumentationCollector("documentation", self.config)
        
        # Wikipedia collector
        if self.config.get("sources", {}).get("wikipedia"):
            self.collectors["wikipedia"] = WikipediaCollector("wikipedia", self.config)
        
        # News collector
        if self.config.get("sources", {}).get("news"):
            self.collectors["news"] = NewsCollector("news", self.config)
        
        # GitHub collector
        if self.config.get("sources", {}).get("github"):
            self.collectors["github"] = GitHubCollector("github", self.config)
        
        self.logger.info(
            "Initialized collectors",
            collector_count=len(self.collectors),
            collectors=list(self.collectors.keys())
        )
    
    async def collect_all_data(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Collect data from all sources.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing collection results
        """
        start_time = time.time()
        
        self.logger.info("Starting data collection from all sources")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "collectors": {},
            "summary": {
                "total_items": 0,
                "successful_collectors": 0,
                "failed_collectors": 0,
                "duration_seconds": 0
            }
        }
        
        # Collect from each source
        for name, collector in self.collectors.items():
            try:
                self.logger.info(f"Collecting from {name}")
                
                # Collect data
                data = await collector.collect()
                
                # Save if requested
                filepath = None
                if save_results and data:
                    filepath = collector.save_data(data)
                
                # Record results
                results["collectors"][name] = {
                    "status": "success",
                    "item_count": len(data),
                    "filepath": filepath,
                    "source_info": collector.get_source_info()
                }
                
                results["summary"]["total_items"] += len(data)
                results["summary"]["successful_collectors"] += 1
                
                self.logger.info(
                    f"Successfully collected from {name}",
                    item_count=len(data),
                    filepath=filepath
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to collect from {name}",
                    error=str(e),
                    exc_info=True
                )
                
                results["collectors"][name] = {
                    "status": "failed",
                    "error": str(e),
                    "item_count": 0
                }
                
                results["summary"]["failed_collectors"] += 1
        
        # Calculate duration
        duration = time.time() - start_time
        results["summary"]["duration_seconds"] = duration
        
        # Log performance
        log_performance(
            "data_collection",
            duration,
            total_items=results["summary"]["total_items"],
            successful_collectors=results["summary"]["successful_collectors"],
            failed_collectors=results["summary"]["failed_collectors"]
        )
        
        self.logger.info(
            "Data collection completed",
            total_items=results["summary"]["total_items"],
            successful_collectors=results["summary"]["successful_collectors"],
            failed_collectors=results["summary"]["failed_collectors"],
            duration_seconds=duration
        )
        
        return results
    
    async def collect_from_source(self, source_name: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Collect data from a specific source.
        
        Args:
            source_name: Name of the source to collect from
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing collection results
        """
        if source_name not in self.collectors:
            raise ValueError(f"Unknown collector: {source_name}")
        
        collector = self.collectors[source_name]
        start_time = time.time()
        
        try:
            self.logger.info(f"Collecting from {source_name}")
            
            # Collect data
            data = await collector.collect()
            
            # Save if requested
            filepath = None
            if save_results and data:
                filepath = collector.save_data(data)
            
            duration = time.time() - start_time
            
            results = {
                "status": "success",
                "source": source_name,
                "item_count": len(data),
                "filepath": filepath,
                "duration_seconds": duration,
                "source_info": collector.get_source_info()
            }
            
            self.logger.info(
                f"Successfully collected from {source_name}",
                item_count=len(data),
                filepath=filepath,
                duration_seconds=duration
            )
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.logger.error(
                f"Failed to collect from {source_name}",
                error=str(e),
                duration_seconds=duration,
                exc_info=True
            )
            
            return {
                "status": "failed",
                "source": source_name,
                "error": str(e),
                "item_count": 0,
                "duration_seconds": duration
            }
    
    def get_collector_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collectors.
        
        Returns:
            Dictionary containing collector statistics
        """
        stats = {
            "total_collectors": len(self.collectors),
            "collectors": {}
        }
        
        for name, collector in self.collectors.items():
            stats["collectors"][name] = collector.get_collection_stats()
        
        return stats
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.
        
        Returns:
            List of available source names
        """
        return list(self.collectors.keys())
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the collector configuration.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if any collectors are configured
        if not self.collectors:
            validation_results["valid"] = False
            validation_results["errors"].append("No data collectors configured")
        
        # Check each collector's configuration
        for name, collector in self.collectors.items():
            try:
                source_info = collector.get_source_info()
                if not source_info:
                    validation_results["warnings"].append(f"No source info for {name}")
            except Exception as e:
                validation_results["errors"].append(f"Configuration error in {name}: {str(e)}")
                validation_results["valid"] = False
        
        return validation_results 